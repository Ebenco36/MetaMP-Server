from __future__ import annotations

import shutil

import pandas as pd

from src.ingestion.context import DatasetSource, IngestionContext
from src.ingestion.http_client import build_retrying_session, default_timeout


class UniProtDatasetSource(DatasetSource):
    def __init__(self, batch_size: int = 200, freshness_days: int = 30):
        self.batch_size = batch_size
        self.freshness_days = freshness_days
        self.session = build_retrying_session(
            headers={
                "Accept": "application/json, text/plain, */*",
            }
        )
        self.search_url = "https://rest.uniprot.org/uniprotkb/search"
        self.uniparc_url = "https://rest.uniprot.org/uniparc/search"

    def run(self, context: IngestionContext):
        existing_dataset = self._reuse_recent_artifacts_if_available(context)
        if existing_dataset is not None:
            return existing_dataset

        mappings = self._load_mappings(context)
        context.report(
            f"[uniprot] Loaded {len(mappings)} unique PDB/UniProt mapping row(s)"
        )
        batch_files = self._fetch_batches(context, mappings)
        context.report(f"[uniprot] Merging {len(batch_files)} batch file(s)")
        master_df = self._merge_batches(context, batch_files)
        context.report(
            f"[uniprot] Writing UniProt dataset outputs with {len(master_df)} row(s)"
        )
        master_df.to_csv(
            context.layout.uniprot_dataset(context.run_date), index=False
        )
        master_df.to_csv(context.layout.uniprot_dataset_current, index=False)
        return master_df

    def _reuse_recent_artifacts_if_available(self, context: IngestionContext):
        latest_dataset = context.layout.latest_recent_dated_path(
            "Uniprot_functions",
            ".csv",
            self.freshness_days,
        )
        current_dataset_path = context.layout.uniprot_dataset_current

        if latest_dataset is None:
            return None
        latest_run_date, dated_dataset_path = latest_dataset

        if not context.layout.uniprot_dataset(context.run_date).exists():
            shutil.copyfile(
                dated_dataset_path,
                context.layout.uniprot_dataset(context.run_date),
            )

        shutil.copyfile(dated_dataset_path, current_dataset_path)

        context.report(
            f"[uniprot] Reusing artifact set from {latest_run_date}; skipping source fetch. "
            f"Adjust INGESTION_ARTIFACT_FRESHNESS_DAYS or delete {dated_dataset_path.name} "
            "and dated batch files under datasets/UniProt to rerun this stage."
        )
        return pd.read_csv(current_dataset_path, low_memory=False)

    def _load_mappings(self, context: IngestionContext):
        data_frame = pd.read_csv(
            context.layout.quantitative_dataset_current,
            usecols=["Pdb Code", "uniprot_id"],
            low_memory=False,
        )
        data_frame = data_frame.dropna(subset=["Pdb Code", "uniprot_id"])
        data_frame = data_frame.drop_duplicates(subset=["Pdb Code", "uniprot_id"])
        return data_frame.rename(columns={"Pdb Code": "pdb_code"}).to_dict(
            orient="records"
        )

    def _fetch_batches(self, context: IngestionContext, mappings):
        batch_files = []
        cache = {}
        total_batches = max(1, (len(mappings) + self.batch_size - 1) // self.batch_size)

        for batch_index, start in enumerate(
            range(0, len(mappings), self.batch_size), start=1
        ):
            batch_rows = mappings[start : start + self.batch_size]
            batch_path = (
                context.layout.uniprot_batch_dir
                / f"Uniprot_functions_{context.run_date}_batch{batch_index}.csv"
            )
            batch_files.append(batch_path)
            if batch_path.exists():
                context.report(
                    f"[uniprot] Reusing batch {batch_index}/{total_batches} from {batch_path.name}"
                )
                continue

            context.report(
                f"[uniprot] Fetching batch {batch_index}/{total_batches} with {len(batch_rows)} mapping(s)"
            )
            records = []
            cache_hits = 0
            fetched_records = 0
            missing_count = 0
            uniparc_fallback_count = 0
            for mapping in batch_rows:
                uniprot_id = str(mapping["uniprot_id"])
                if uniprot_id in cache:
                    cache_hits += 1
                else:
                    cache[uniprot_id] = self._fetch_uniprot_record(uniprot_id)
                    fetched_records += 1

                source_record = cache[uniprot_id]
                if source_record is None:
                    missing_count += 1
                    continue
                if source_record.get("data_source") == "UniParc":
                    uniparc_fallback_count += 1

                record = dict(source_record)
                record["uniprot_id"] = uniprot_id
                record["pdb_code"] = mapping["pdb_code"]
                records.append(record)

            if records:
                pd.DataFrame(records).to_csv(batch_path, index=False)
                context.report(
                    "[uniprot] Completed batch "
                    f"{batch_index}/{total_batches} with {len(records)} record(s), "
                    f"{fetched_records} upstream fetch(es), {cache_hits} cache hit(s), "
                    f"{uniparc_fallback_count} UniParc fallback(s), {missing_count} missing mapping(s)"
                )
            else:
                context.report(
                    f"[uniprot] Batch {batch_index}/{total_batches} produced no records"
                )

        return batch_files

    def _fetch_uniprot_record(self, uniprot_id):
        response = self.session.get(
            self.search_url,
            params={"query": f"accession:{uniprot_id}", "format": "json"},
            timeout=default_timeout(),
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return self._extract_information(results[0], source="uniprotKB")
        elif response.status_code != 404:
            response.raise_for_status()

        response = self.session.get(
            self.uniparc_url,
            params={"query": uniprot_id, "format": "json"},
            timeout=default_timeout(),
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()

        results = response.json().get("results", [])
        if not results:
            return None
        return self._extract_information(results[0], source="UniParc")

    def _extract_information(self, data: dict, source: str):
        record = {
            "data_source": source,
            "uniprot_id": "",
            "pdb_code": "",
            "uniProtkb_id": "",
            "info_type": "",
            "info_created": "",
            "info_modified": "",
            "info_sequence_update": "",
            "annotation_score": "",
            "taxon_id": "",
            "organism_scientific_name": "",
            "organism_common_name": "",
            "organism_lineage": "",
            "secondary_accession": "",
            "protein_recommended_name": "",
            "protein_alternative_name": "",
            "associated_genes": "",
            "comment_function": [],
            "comment_interactions": [],
            "comment_catalytic_activity": [],
            "comment_subunit": [],
            "comment_PTM": [],
            "comment_caution": [],
            "comment_tissue_specificity": [],
            "comment_subcellular_locations": [],
            "comment_alternative_products": [],
            "comment_disease_name": "",
            "comment_disease": [],
            "comment_similarity": [],
            "features": [],
            "references": [],
            "cross_references": [],
            "keywords": [],
            "sequence_length": 0,
            "sequence_mass": 0,
            "sequence_sequence": "",
            "extra_attributes": [],
            "molecular_function": "",
            "cellular_component": "",
            "biological_process": "",
        }

        if source == "uniprotKB":
            record["uniprot_id"] = data.get("primaryAccession", "")
            record["uniProtkb_id"] = data.get("uniProtkbId", "")
            record["info_type"] = data.get("entryType", "")
            audit = data.get("entryAudit", {})
            record["info_created"] = audit.get("firstPublicDate", "")
            record["info_modified"] = audit.get("lastAnnotationUpdateDate", "")
            record["info_sequence_update"] = audit.get(
                "lastSequenceUpdateDate", ""
            )
            record["annotation_score"] = data.get("annotationScore", "")
            organism = data.get("organism", {})
            record["taxon_id"] = organism.get("taxonId", "")
            record["organism_scientific_name"] = organism.get("scientificName", "")
            record["organism_common_name"] = organism.get("commonName", "")
            record["organism_lineage"] = ";".join(organism.get("lineage", []))
            record["secondary_accession"] = ";".join(
                data.get("secondaryAccessions", [])
            )
            protein_description = data.get("proteinDescription", {})
            record["protein_recommended_name"] = (
                protein_description.get("recommendedName", {})
                .get("fullName", {})
                .get("value", "")
            )
            record["protein_alternative_name"] = ";".join(
                [
                    item.get("fullName", {}).get("value", "")
                    for item in protein_description.get("alternativeNames", [])
                ]
            )
            record["associated_genes"] = ";".join(
                [
                    gene.get("geneName", {}).get("value", "")
                    for gene in data.get("genes", [])
                ]
            )
            for comment in data.get("comments", []):
                comment_type = comment.get("commentType", "")
                if comment_type == "FUNCTION":
                    record["comment_function"] = comment.get("texts", [])
                if comment_type == "INTERACTION":
                    record["comment_interactions"] = comment.get("interactions", [])
                if comment_type == "SUBCELLULAR LOCATION":
                    record["comment_subcellular_locations"] = comment.get(
                        "subcellularLocations", []
                    )
                if comment_type == "ALTERNATIVE PRODUCTS":
                    record["comment_alternative_products"] = comment.get(
                        "isoforms", []
                    )
                if comment_type == "DISEASE":
                    record["comment_disease"].append(comment.get("disease", {}))
                if comment_type == "PTM":
                    record["comment_PTM"] = comment.get("texts", [])
                if comment_type == "CATALYTIC ACTIVITY":
                    record["comment_catalytic_activity"].append(
                        comment.get("reaction", {})
                    )

            record["comment_disease_name"] = ";".join(
                [
                    disease.get("diseaseId", "")
                    for disease in record["comment_disease"]
                ]
            )
            record["features"] = data.get("features", [])
            record["references"] = data.get("references", [])
            record["keywords"] = data.get("keywords", [])
            sequence = data.get("sequence", {})
            record["sequence_length"] = sequence.get("length", 0)
            record["sequence_mass"] = sequence.get("molWeight", 0)
            record["sequence_sequence"] = sequence.get("value", "")
            record["extra_attributes"] = data.get("uniProtKBCrossReferences", [])

            go_mapping = {
                "F": "molecular_function",
                "C": "cellular_component",
                "P": "biological_process",
            }
            for cross_reference in data.get("uniProtKBCrossReferences", []):
                for prop in cross_reference.get("properties", []):
                    value = prop.get("value", "")
                    if ":" in value and len(value) > 2 and value[1] == ":":
                        key, term = value.split(":", 1)
                        field_name = go_mapping.get(key)
                        if field_name:
                            record[field_name] += f"{term};"

            for field_name in [
                "molecular_function",
                "cellular_component",
                "biological_process",
            ]:
                record[field_name] = record[field_name].strip(";")
        else:
            record["uniprot_id"] = data.get("uniParcId", "")
            record["info_created"] = data.get("oldestCrossRefCreated", "")
            record["info_modified"] = data.get("mostRecentCrossRefUpdated", "")
            record["cross_references"] = data.get("uniParcCrossReferences", [])

        return record

    def _merge_batches(self, context: IngestionContext, batch_files):
        data_frames = [
            pd.read_csv(batch_file, low_memory=False)
            for batch_file in batch_files
            if batch_file.exists()
        ]
        if not data_frames:
            raise FileNotFoundError("No UniProt batch files were produced.")

        merged = pd.concat(data_frames, ignore_index=True)
        context.report(
            f"[uniprot] Merged batch files into {len(merged)} total row(s)"
        )
        return merged
