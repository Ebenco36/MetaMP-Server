from __future__ import annotations

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
        pdb_code = protein[0].text
        member_proteins = []
        member_entries = []

        for memberprotein in protein[10]:
            member_proteins.append([memberprotein[0].tag, memberprotein[0].text])
            member_entries.append(
                [
                    group_name,
                    subgroup_name,
                    memberprotein[0].text,
                    memberprotein[1].text,
                    memberprotein[2].text,
                    memberprotein[3].text,
                    memberprotein[4].text,
                    memberprotein[5].text,
                    memberprotein[6].text,
                    memberprotein[7].text,
                    [
                        [memberprotein[8][index].tag, memberprotein[8][index].text]
                        for index in range(len(memberprotein[8]))
                    ],
                    memberprotein[9].text,
                    memberprotein[10].text,
                    None,
                ]
            )

        master_entry = [
            group_name,
            subgroup_name,
            pdb_code,
            "MasterProtein",
            protein[1].text,
            protein[2].text,
            protein[3].text,
            protein[4].text,
            protein[5].text,
            protein[6].text,
            [[protein[7][index].tag, protein[7][index].text] for index in range(len(protein[7]))],
            protein[8].text,
            protein[9].text,
            member_proteins,
        ]
        return master_entry, member_entries
