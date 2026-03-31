from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as element_tree
from pathlib import Path

import pandas as pd
from requests import RequestException

from src.ingestion.context import DatasetSource, IngestionContext
from src.ingestion.http_client import build_retrying_session, default_timeout


class MPStrucDatasetSource(DatasetSource):
    XML_URL = "https://blanco.biomol.uci.edu/mpstruc/listAll/mpstrucTblXml"

    def __init__(self, allow_stale_fallback: bool = True, freshness_days: int = 30):
        self.session = build_retrying_session()
        self.allow_stale_fallback = allow_stale_fallback
        self.freshness_days = freshness_days

    def run(self, context: IngestionContext):
        context.layout.ensure_directories()
        xml_path = context.layout.mpstruc_xml(context.run_date)

        if os.getenv("MPSTRUC_REPARSE_CURRENT_XML", "false").lower() == "true":
            reparsed = self._reparse_current_xml_snapshot(context, xml_path)
            if reparsed is not None:
                return reparsed

        existing_dataset = self._reuse_recent_artifacts_if_available(context, xml_path)
        if existing_dataset is not None:
            return existing_dataset

        context.report(f"[mpstruc] Downloading XML source to {xml_path.name}")

        try:
            with self.session.get(
                self.XML_URL,
                stream=True,
                timeout=default_timeout(),
            ) as response:
                response.raise_for_status()
                with xml_path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            output.write(chunk)
        except RequestException as error:
            if not self.allow_stale_fallback:
                raise
            return self._reuse_current_snapshot(context, xml_path, error)

        context.report("[mpstruc] XML download complete, updating current snapshot")
        context.layout.mpstruc_xml_current.write_bytes(xml_path.read_bytes())
        context.report("[mpstruc] Parsing XML into normalized dataset rows")
        dataset, ids = self._parse_xml(xml_path)
        dataset.to_csv(
            context.layout.mpstruc_dataset(context.run_date), index=False
        )
        dataset.to_csv(context.layout.mpstruc_dataset_current, index=False)
        ids.to_csv(context.layout.mpstruc_ids_current, index=False)
        context.report(
            f"[mpstruc] Wrote {len(dataset)} row(s) and {len(ids)} identifier(s)"
        )
        return dataset

    def _reparse_current_xml_snapshot(self, context: IngestionContext, xml_path):
        current_xml_path = context.layout.mpstruc_xml_current
        latest_xml_snapshot = current_xml_path if current_xml_path.exists() else self._find_latest_local_xml_snapshot(context.layout)
        if latest_xml_snapshot is None:
            context.report("[mpstruc] MPSTRUC_REPARSE_CURRENT_XML requested, but no local XML snapshot is available")
            return None

        context.report(
            f"[mpstruc] Reparsing local XML snapshot {latest_xml_snapshot.name} without downloading from source"
        )
        if latest_xml_snapshot != xml_path:
            self._try_write_bytes(
                xml_path,
                latest_xml_snapshot.read_bytes(),
                context,
                "[mpstruc] Unable to materialize dated XML snapshot from local XML seed",
            )
        if latest_xml_snapshot != current_xml_path:
            self._try_write_bytes(
                current_xml_path,
                latest_xml_snapshot.read_bytes(),
                context,
                "[mpstruc] Unable to refresh current XML snapshot from local XML seed",
            )

        dataset, ids = self._parse_xml(latest_xml_snapshot)
        dataset.to_csv(context.layout.mpstruc_dataset(context.run_date), index=False)
        dataset.to_csv(context.layout.mpstruc_dataset_current, index=False)
        ids.to_csv(context.layout.mpstruc_ids_current, index=False)
        context.report(
            f"[mpstruc] Rebuilt {len(dataset)} row(s) and {len(ids)} identifier(s) from local XML snapshot"
        )
        return dataset

    def _reuse_recent_artifacts_if_available(self, context: IngestionContext, xml_path):
        latest_xml = context.layout.latest_recent_dated_path(
            "mpstrucTblXml",
            ".xml",
            self.freshness_days,
        )
        latest_dataset = context.layout.latest_recent_dated_path(
            "Mpstruct_dataset",
            ".csv",
            self.freshness_days,
        )
        current_dataset_path = context.layout.mpstruc_dataset_current
        current_ids_path = context.layout.mpstruc_ids_current
        current_xml_path = context.layout.mpstruc_xml_current

        if latest_xml is None or latest_dataset is None or current_ids_path.exists() is False:
            return None

        latest_run_date, latest_xml_path = latest_xml
        latest_dataset_run_date, latest_dataset_path = latest_dataset
        if latest_run_date != latest_dataset_run_date:
            return None

        if not xml_path.exists():
            self._try_copyfile(
                latest_xml_path,
                xml_path,
                context,
                "[mpstruc] Unable to materialize dated XML snapshot from the latest reusable artifact",
            )
        if not context.layout.mpstruc_dataset(context.run_date).exists():
            self._try_copyfile(
                latest_dataset_path,
                context.layout.mpstruc_dataset(context.run_date),
                context,
                "[mpstruc] Unable to materialize dated dataset snapshot from the latest reusable artifact",
            )

        self._try_copyfile(
            latest_dataset_path,
            current_dataset_path,
            context,
            "[mpstruc] Unable to refresh current dataset snapshot from the latest reusable artifact",
        )
        self._try_copyfile(
            latest_xml_path,
            current_xml_path,
            context,
            "[mpstruc] Unable to refresh current XML snapshot from the latest reusable artifact",
        )

        dataset = pd.read_csv(current_dataset_path, low_memory=False)
        dataset["Pdb Code"].to_csv(current_ids_path, index=False)
        context.report(
            f"[mpstruc] Reusing artifact set from {latest_run_date}; skipping source fetch. "
            f"Adjust INGESTION_ARTIFACT_FRESHNESS_DAYS or delete {latest_xml_path.name} "
            f"and {latest_dataset_path.name} to rerun this stage."
        )
        return dataset

    def _reuse_current_snapshot(self, context: IngestionContext, xml_path, error):
        current_dataset_path = context.layout.mpstruc_dataset_current
        current_ids_path = context.layout.mpstruc_ids_current
        current_xml_path = context.layout.mpstruc_xml_current

        if current_dataset_path.exists() and current_ids_path.exists():
            context.report(
                "[mpstruc] Source download timed out; reusing the current local MPStruc snapshot"
            )

            if current_xml_path.exists():
                self._try_write_bytes(
                    xml_path,
                    current_xml_path.read_bytes(),
                    context,
                    "[mpstruc] Unable to materialize dated XML snapshot from current XML fallback",
                )

            dated_dataset_path = context.layout.mpstruc_dataset(context.run_date)
            self._try_write_bytes(
                dated_dataset_path,
                current_dataset_path.read_bytes(),
                context,
                "[mpstruc] Unable to materialize dated dataset snapshot from current dataset fallback",
            )

            dataset = pd.read_csv(current_dataset_path, low_memory=False)
            ids = pd.read_csv(current_ids_path, low_memory=False)
            context.report(
                f"[mpstruc] Reused {len(dataset)} row(s) and {len(ids)} identifier(s) from current snapshot"
            )
            return dataset

        latest_xml_snapshot = self._find_latest_local_xml_snapshot(context.layout)
        if latest_xml_snapshot is None:
            raise RuntimeError(
                "MPStruc download failed and no current snapshot is available for fallback."
            ) from error

        context.report(
            "[mpstruc] Source download timed out; rebuilding current MPStruc snapshot from the latest local XML seed"
        )
        if latest_xml_snapshot != xml_path:
            self._try_write_bytes(
                xml_path,
                latest_xml_snapshot.read_bytes(),
                context,
                "[mpstruc] Unable to materialize dated XML snapshot from the latest local XML seed",
            )
        self._try_write_bytes(
            current_xml_path,
            latest_xml_snapshot.read_bytes(),
            context,
            "[mpstruc] Unable to refresh current XML snapshot from the latest local XML seed",
        )

        dataset, ids = self._parse_xml(latest_xml_snapshot)
        dataset.to_csv(context.layout.mpstruc_dataset(context.run_date), index=False)
        dataset.to_csv(current_dataset_path, index=False)
        ids.to_csv(current_ids_path, index=False)
        context.report(
            f"[mpstruc] Rebuilt {len(dataset)} row(s) and {len(ids)} identifier(s) from local XML snapshot {latest_xml_snapshot.name}"
        )
        return dataset

    @staticmethod
    def _try_write_bytes(target: Path, payload: bytes, context: IngestionContext, warning: str):
        try:
            target.write_bytes(payload)
        except PermissionError:
            context.report(f"{warning}; continuing with available source file")

    @staticmethod
    def _try_copyfile(source: Path, target: Path, context: IngestionContext, warning: str):
        try:
            shutil.copyfile(source, target)
        except PermissionError:
            context.report(f"{warning}; continuing with available source file")

    @staticmethod
    def _find_latest_local_xml_snapshot(layout) -> Path | None:
        candidates = sorted(
            {
                path
                for path in layout.base_dir.glob("mpstrucTblXml*.xml")
                if path.exists() and path.is_file()
            },
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _parse_xml(self, xml_path):
        tree = element_tree.parse(xml_path)
        root = tree.getroot()
        protein_entries = []

        for groups in root:
            for group in groups:
                group_name = group[0].text
                for subgroup in group[2]:
                    subgroup_name = subgroup[0].text
                    for protein in subgroup[1]:
                        (
                            master_entry,
                            member_entries,
                        ) = self._extract_protein_entries(
                            protein,
                            group_name,
                            subgroup_name,
                        )
                        protein_entries.append(master_entry)
                        protein_entries.extend(member_entries)

        columns = [
            "Group",
            "Subgroup",
            "Pdb Code",
            "Is Master Protein?",
            "Name",
            "Species",
            "Taxonomic Domain",
            "Expressed in Species",
            "Resolution",
            "Description",
            "Bibliography",
            "Secondary Bibliogrpahies",
            "Related Pdb Entries",
            "Member Proteins",
        ]
        dataset = pd.DataFrame(protein_entries, columns=columns)
        ids = dataset["Pdb Code"]
        return dataset, ids

    def _extract_protein_entries(self, protein, group_name, subgroup_name):
        pdb_code = self._find_text(protein, "pdbCode")
        member_proteins = []
        member_entries = []

        member_protein_container = protein.find("memberProteins")
        for memberprotein in list(member_protein_container or []):
            member_pdb_code = self._find_text(memberprotein, "pdbCode")
            member_proteins.append(["pdbCode", member_pdb_code])
            member_entries.append(
                [
                    group_name,
                    subgroup_name,
                    member_pdb_code,
                    self._find_text(memberprotein, "masterProteinPdbCode"),
                    self._find_text(memberprotein, "name"),
                    self._find_text(memberprotein, "species"),
                    self._find_text(memberprotein, "taxonomicDomain"),
                    self._find_text(memberprotein, "expressedInSpecies"),
                    self._find_text(memberprotein, "resolution"),
                    self._find_text(memberprotein, "description"),
                    self._extract_tag_text_pairs(memberprotein.find("bibliography")),
                    self._find_text(memberprotein, "secondaryBibliographies"),
                    self._extract_related_pdb_entries(memberprotein),
                    None,
                ]
            )

        master_entry = [
            group_name,
            subgroup_name,
            pdb_code,
            "MasterProtein",
            self._find_text(protein, "name"),
            self._find_text(protein, "species"),
            self._find_text(protein, "taxonomicDomain"),
            self._find_text(protein, "expressedInSpecies"),
            self._find_text(protein, "resolution"),
            self._find_text(protein, "description"),
            self._extract_tag_text_pairs(protein.find("bibliography")),
            self._find_text(protein, "secondaryBibliographies"),
            self._extract_related_pdb_entries(protein),
            member_proteins,
        ]
        return master_entry, member_entries

    @staticmethod
    def _find_text(parent, tag_name, default=""):
        if parent is None:
            return default
        node = parent.find(tag_name)
        if node is None or node.text is None:
            return default
        text = str(node.text).strip()
        return text if text else default

    @staticmethod
    def _extract_tag_text_pairs(container):
        if container is None:
            return []
        pairs = []
        for child in list(container):
            pairs.append([child.tag, (child.text or "").strip() or None])
        return pairs

    def _extract_related_pdb_entries(self, parent):
        container = parent.find("relatedPdbEntries") if parent is not None else None
        if container is None:
            return json.dumps([])

        related_codes = []
        for child in list(container):
            value = (child.text or "").strip()
            if value:
                related_codes.append(value)

        return json.dumps(related_codes)
