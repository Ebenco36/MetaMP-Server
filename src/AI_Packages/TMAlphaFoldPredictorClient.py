import json
from dataclasses import dataclass
from typing import Iterable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.AI_Packages.TMProteinPredictor import serialize_tm_regions


TMALPHAFOLD_PROVIDER = "TMAlphaFold"
TMALPHAFOLD_BASE_URL = "https://tmalphafold.ttk.hu/api"
TMALPHAFOLD_SEQUENCE_METHODS = (
    "DeepTMHMM",
    "Hmmtop",
    "Memsat",
    "Octopus",
    "Philius",
    "Phobius",
    "Pro",
    "Prodiv",
    "Scampi",
    "ScampiMsa",
    "TMHMM",
    "Topcons2",
)
TMALPHAFOLD_AUX_METHODS = ("SignalP",)
TMALPHAFOLD_ALL_METHODS = TMALPHAFOLD_SEQUENCE_METHODS + TMALPHAFOLD_AUX_METHODS
TMALPHAFOLD_MEMBRANE_LABELS = {"membrane", "tmhelix", "transmembrane", "tm"}
TMALPHAFOLD_TMDET_MEMBRANE_TYPES = {"M"}


@dataclass
class TMAlphaFoldPredictionResult:
    pdb_code: str
    uniprot_id: str
    provider: str
    method: str
    prediction_kind: str
    tm_count: Optional[int]
    tm_regions_json: str
    raw_payload_json: str
    source_url: str
    status: str
    sequence_sequence: Optional[str] = None
    error_message: Optional[str] = None


def _normalize_method_name(method: str) -> str:
    normalized = str(method or "").strip()
    for candidate in TMALPHAFOLD_ALL_METHODS:
        if candidate.lower() == normalized.lower():
            return candidate
    raise ValueError(
        f"Unsupported TMAlphaFold method '{method}'. "
        f"Expected one of: {', '.join(TMALPHAFOLD_ALL_METHODS)}."
    )


def _normalize_tm_regions(regions: Iterable[dict]) -> list[dict]:
    normalized = []
    for index, region in enumerate(regions, start=1):
        try:
            start = max(1, int(region["start"]))
            end = max(1, int(region["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        region_index = _scalar_value(region.get("index"))
        try:
            region_index = int(region_index) if region_index is not None else index
        except (TypeError, ValueError):
            region_index = index
        item = {
            "index": region_index,
            "start": start,
            "end": end,
            "length": end - start + 1,
        }
        if region.get("label"):
            item["label"] = region["label"]
        normalized.append(item)
    return normalized


def _listify_region_payload(region_payload):
    if region_payload is None:
        return []
    if isinstance(region_payload, list):
        return region_payload
    if isinstance(region_payload, tuple):
        return list(region_payload)
    if isinstance(region_payload, dict):
        return [region_payload]
    if hasattr(region_payload, "tolist"):
        normalized = region_payload.tolist()
        if isinstance(normalized, list):
            return normalized
        if isinstance(normalized, dict):
            return [normalized]
    return []


def _scalar_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _scalar_value(value[0])
    if hasattr(value, "tolist"):
        return _scalar_value(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return None
    return value


def _string_value(*candidates) -> str:
    for candidate in candidates:
        scalar = _scalar_value(candidate)
        if scalar is None:
            continue
        text = str(scalar).strip()
        if text:
            return text
    return ""


def _extract_sequence_from_chain_payload(payload) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    chains = _listify_region_payload(payload.get("CHAIN"))
    for chain in chains:
        if not isinstance(chain, dict):
            continue
        sequence = _string_value(chain.get("SEQ"))
        if sequence:
            return sequence
    return None


def _parse_regions_from_payload(payload, prediction_kind: str) -> list[dict]:
    regions_root = payload.get("Regions") if isinstance(payload, dict) else None
    if not isinstance(regions_root, dict):
        return []
    region_entries = _listify_region_payload(regions_root.get("Region"))
    normalized_regions = []
    for index, region_entry in enumerate(region_entries, start=1):
        attributes = region_entry.get("_attributes") if isinstance(region_entry, dict) else None
        if not isinstance(attributes, dict):
            continue
        loc = _string_value(attributes.get("Loc"), attributes.get("Label"))
        start = _scalar_value(attributes.get("Start"))
        end = _scalar_value(attributes.get("End"))
        normalized_regions.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "label": loc or prediction_kind,
                "attributes": attributes,
            }
        )
    return _normalize_tm_regions(normalized_regions)


def _parse_tmdet_regions_from_payload(payload) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    chains = _listify_region_payload(payload.get("CHAIN"))
    normalized_regions = []
    region_index = 1
    for chain in chains:
        if not isinstance(chain, dict):
            continue
        chain_id = _string_value((chain.get("_attributes") or {}).get("CHAINID"))
        regions = _listify_region_payload(chain.get("REGION"))
        for region in regions:
            attributes = region.get("_attributes") if isinstance(region, dict) else None
            if not isinstance(attributes, dict):
                continue
            region_type = _string_value(attributes.get("type")).upper()
            if region_type not in TMALPHAFOLD_TMDET_MEMBRANE_TYPES:
                continue
            normalized_regions.append(
                {
                    "index": region_index,
                    "start": _scalar_value(attributes.get("seq_beg")),
                    "end": _scalar_value(attributes.get("seq_end")),
                    "label": f"TMDET membrane{f' chain {chain_id}' if chain_id else ''}",
                }
            )
            region_index += 1
    return _normalize_tm_regions(normalized_regions)


def _count_transmembrane_regions(regions: Iterable[dict]) -> int:
    count = 0
    for region in regions or ():
        label = _string_value(region.get("label")).lower()
        if label in TMALPHAFOLD_MEMBRANE_LABELS:
            count += 1
    return count


def _prediction_kind_for_method(method: str) -> str:
    return "signal_peptide" if method == "SignalP" else "sequence_topology"


def _extract_sequence_from_payload(payload) -> Optional[str]:
    sequence_root = payload.get("Sequence") if isinstance(payload, dict) else None
    if not isinstance(sequence_root, dict):
        return _extract_sequence_from_chain_payload(payload)
    sequence = _string_value(sequence_root.get("Seq"))
    return sequence or _extract_sequence_from_chain_payload(payload)


class TMAlphaFoldPredictorClient:
    def __init__(self, base_url: str = TMALPHAFOLD_BASE_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        retries = Retry(
            total=4,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _fetch_json(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def fetch_method(self, pdb_code: str, uniprot_id: str, method: str) -> TMAlphaFoldPredictionResult:
        normalized_method = _normalize_method_name(method)
        prediction_kind = _prediction_kind_for_method(normalized_method)
        path = f"/prediction/{normalized_method}/{uniprot_id}.json"
        source_url = f"{self.base_url}{path}"
        try:
            payload = self._fetch_json(path)
            regions = _parse_regions_from_payload(payload, prediction_kind)
            tm_count = (
                _count_transmembrane_regions(regions)
                if prediction_kind == "sequence_topology"
                else None
            )
            return TMAlphaFoldPredictionResult(
                pdb_code=pdb_code,
                uniprot_id=uniprot_id,
                provider=TMALPHAFOLD_PROVIDER,
                method=normalized_method,
                prediction_kind=prediction_kind,
                tm_count=tm_count,
                tm_regions_json=serialize_tm_regions(regions),
                raw_payload_json=json.dumps(payload),
                source_url=source_url,
                status="success",
                sequence_sequence=_extract_sequence_from_payload(payload),
            )
        except Exception as exc:
            return TMAlphaFoldPredictionResult(
                pdb_code=pdb_code,
                uniprot_id=uniprot_id,
                provider=TMALPHAFOLD_PROVIDER,
                method=normalized_method,
                prediction_kind=prediction_kind,
                tm_count=None,
                tm_regions_json="[]",
                raw_payload_json="",
                source_url=source_url,
                status="error",
                sequence_sequence=None,
                error_message=str(exc),
            )

    def fetch_tmdet(self, pdb_code: str, uniprot_id: str) -> TMAlphaFoldPredictionResult:
        path = f"/tmdet/{uniprot_id}.json"
        source_url = f"{self.base_url}{path}"
        try:
            payload = self._fetch_json(path)
            regions = _parse_tmdet_regions_from_payload(payload)
            tm_count = len(regions) if regions else None
            return TMAlphaFoldPredictionResult(
                pdb_code=pdb_code,
                uniprot_id=uniprot_id,
                provider=TMALPHAFOLD_PROVIDER,
                method="TMDET",
                prediction_kind="structure_membrane_plane",
                tm_count=tm_count,
                tm_regions_json=serialize_tm_regions(regions),
                raw_payload_json=json.dumps(payload),
                source_url=source_url,
                status="success",
                sequence_sequence=_extract_sequence_from_payload(payload),
            )
        except Exception as exc:
            return TMAlphaFoldPredictionResult(
                pdb_code=pdb_code,
                uniprot_id=uniprot_id,
                provider=TMALPHAFOLD_PROVIDER,
                method="TMDET",
                prediction_kind="structure_membrane_plane",
                tm_count=None,
                tm_regions_json="[]",
                raw_payload_json="",
                source_url=source_url,
                status="error",
                sequence_sequence=None,
                error_message=str(exc),
            )
