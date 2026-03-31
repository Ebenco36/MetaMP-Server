import subprocess
import sys
import re
import os
import time
import traceback
import json
import hashlib
import logging
import shutil
import site
from functools import lru_cache
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd
import psycopg2
from psycopg2 import sql
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor
try:
    import torch
except ImportError:  # pragma: no cover - exercised in non-ML runtime images
    torch = None


logger = logging.getLogger("tmbed_runner")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

_TMBED_UTILS_PATCH_ATTEMPTED = False


def _is_missing_value(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, dict):
        return len(value) == 0
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    try:
        missing = pd.isna(value)
        if isinstance(missing, (list, tuple)):
            return all(bool(item) for item in missing)
        if hasattr(missing, "all") and not isinstance(missing, bool):
            return bool(missing.all())
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _safe_optional_int(value):
    if _is_missing_value(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _has_meaningful_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    try:
        missing = pd.isna(value)
        if isinstance(missing, (list, tuple)):
            return not all(bool(item) for item in missing)
        if hasattr(missing, "all") and not isinstance(missing, bool):
            return not bool(missing.all())
        return not bool(missing)
    except (TypeError, ValueError):
        return True


def _resolve_runtime_directory_path(configured_path: str, local_relative_path: str) -> Path:
    candidate = Path(configured_path)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except (FileNotFoundError, PermissionError, OSError):
        if str(candidate).startswith("/var/app"):
            fallback = Path.cwd() / local_relative_path
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        raise


def _get_tmbed_model_dir() -> Path:
    model_dir = os.getenv("TMBED_MODEL_DIR", "/var/app/data/tmbed-models")
    return _resolve_runtime_directory_path(model_dir, "data/tmbed-models")


def _find_tmbed_utils_path() -> Optional[Path]:
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    try:
        import tmbed

        candidates.append(str(Path(tmbed.__file__).parent.parent))
    except Exception:
        pass

    for site_path in candidates:
        candidate = Path(site_path) / "tmbed" / "utils.py"
        if candidate.exists():
            return candidate
    return None


def _patch_tmbed_utils() -> None:
    global _TMBED_UTILS_PATCH_ATTEMPTED
    if _TMBED_UTILS_PATCH_ATTEMPTED:
        return
    _TMBED_UTILS_PATCH_ATTEMPTED = True

    utils_path = _find_tmbed_utils_path()
    if utils_path is None:
        logger.warning("[Patch] tmbed/utils.py not found; skipping patch.")
        return

    try:
        lines = utils_path.read_text().splitlines(keepends=True)
    except OSError as exc:
        logger.warning(
            "[Patch] Unable to read %s; continuing without patch: %s",
            utils_path,
            exc,
        )
        return
    full_text = "".join(lines)
    if "idx = 0  # patch:" in full_text and "if proteins:" in full_text:
        logger.info(f"[Patch] Already patched: {utils_path}")
        return

    buggy_line_idx = None
    for index, line in enumerate(lines):
        if "batches.append((last_idx, idx + 1))" in line:
            buggy_line_idx = index
            break

    if buggy_line_idx is None:
        return

    buggy_line = lines[buggy_line_idx]
    indent_count = len(buggy_line) - len(buggy_line.lstrip())
    pad = " " * indent_count

    for_loop_idx = None
    for probe in range(buggy_line_idx - 1, max(0, buggy_line_idx - 30), -1):
        if "for idx," in lines[probe] and "enumerate" in lines[probe]:
            for_loop_idx = probe
            break

    backup = utils_path.with_suffix(".py.bak")
    try:
        if not backup.exists():
            shutil.copy2(utils_path, backup)
            logger.info(f"[Patch] Backup -> {backup}")
    except OSError as exc:
        logger.info(
            "[Patch] Unable to create writable backup for %s; assuming packaged patch is sufficient: %s",
            utils_path,
            exc,
        )
        return

    if for_loop_idx is not None:
        loop_line = lines[for_loop_idx]
        loop_indent = len(loop_line) - len(loop_line.lstrip())
        loop_pad = " " * loop_indent
        init_line = f"{loop_pad}idx = 0  # patch: fix UnboundLocalError\n"
        if "idx = 0" not in lines[max(0, for_loop_idx - 1)]:
            lines.insert(for_loop_idx, init_line)
            buggy_line_idx += 1

    lines[buggy_line_idx] = (
        f"{pad}if proteins:\n"
        f"{pad}    batches.append((last_idx, idx + 1))\n"
    )
    try:
        utils_path.write_text("".join(lines))
    except OSError as exc:
        logger.warning(
            "[Patch] Unable to write patched TMbed utils to %s; continuing without patch: %s",
            utils_path,
            exc,
        )
        return
    logger.info(f"[Patch] Successfully patched: {utils_path}")


def _fasta_hash(fasta_text: str) -> str:
    return hashlib.md5(fasta_text.strip().encode()).hexdigest()


def _cache_is_valid(fasta_text: str, embeddings_path: Path) -> bool:
    hash_path = embeddings_path.with_suffix(".hash")
    if not embeddings_path.exists() or not hash_path.exists():
        return False
    is_valid = hash_path.read_text().strip() == _fasta_hash(fasta_text)
    if not is_valid:
        logger.info("[Cache] FASTA changed; stale embeddings will be regenerated.")
    return is_valid


def _write_cache_hash(fasta_text: str, embeddings_path: Path) -> None:
    embeddings_path.with_suffix(".hash").write_text(_fasta_hash(fasta_text))


def detect_tmbed_device(use_gpu: bool = True) -> str:
    if torch is None:
        logger.info("[Device] CPU (torch runtime unavailable)")
        return "cpu"
    if use_gpu and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"[Device] CUDA GPU: {name} ({vram:.1f} GB VRAM)")
        return "cuda"
    if use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("[Device] Apple MPS (Metal Performance Shaders)")
        return "mps"
    logger.info(f"[Device] CPU ({torch.get_num_threads()} threads)")
    return "cpu"


def _build_tmbed_env(device: str, model_dir: Path) -> dict:
    env = os.environ.copy()
    env.setdefault("TMBED_MODEL_DIR", str(model_dir))
    env.setdefault("HF_HOME", str(model_dir / "hf-home"))
    env.setdefault("XDG_CACHE_HOME", str(model_dir / ".cache"))
    env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device == "mps":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


def _tmbed_gpu_flags(device: str, cpu_fallback: bool = True) -> list[str]:
    if device in {"cuda", "mps"}:
        return ["--use-gpu", "--cpu-fallback" if cpu_fallback else "--no-cpu-fallback"]
    return ["--no-use-gpu"]


def _tmbed_thread_flags(device: str, threads: Optional[int]) -> list[str]:
    if (
        device == "cpu"
        and threads is not None
        and _tmbed_subcommand_supports_option("embed", "--threads")
        and _tmbed_subcommand_supports_option("predict", "--threads")
    ):
        return ["--threads", str(threads)]
    return []


@lru_cache(maxsize=8)
def _tmbed_subcommand_supports_option(subcommand: str, option_name: str) -> bool:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "tmbed", subcommand, "--help"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return False
    help_text = "\n".join(
        part for part in [completed.stdout or "", completed.stderr or ""] if part
    )
    return option_name in help_text


def _candidate_tmbed_batch_sizes(batch_size: int) -> list[int]:
    start = max(1, int(batch_size or 1))
    candidates = [start]
    while start > 1:
        start = max(1, start // 2)
        candidates.append(start)
        if start == 1:
            break
    return list(dict.fromkeys(candidates))


def _tmbed_result_looks_resource_killed(result: subprocess.CompletedProcess) -> bool:
    if int(result.returncode or 0) == -9:
        return True
    combined = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    return "killed" in combined or "out of memory" in combined


def _ensure_tmbed_model_downloaded(model_dir: Path, device: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    marker_path = model_dir / ".download_complete"
    if marker_path.exists():
        return
    logger.info("[TMbed] Downloading ProtT5 model (~2.25 GB)...")
    cmd = [sys.executable, "-m", "tmbed", "download"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_build_tmbed_env(device, model_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(f"TMbed download failed:\n{result.stderr}")
    marker_path.write_text("ok\n")
    logger.info("[TMbed] Model ready.")


def _run_tmbed_embed(
    fasta_path: str,
    embeddings_path: Path,
    device: str,
    batch_size: int,
    threads: Optional[int],
    model_dir: Path,
    cpu_fallback: bool = True,
) -> None:
    last_result = None
    for attempt_batch_size in _candidate_tmbed_batch_sizes(batch_size):
        logger.info(
            f"[TMbed embed] device={device} batch_size={attempt_batch_size}"
        )
        cmd = [
            sys.executable,
            "-m",
            "tmbed",
            "embed",
            "-f",
            fasta_path,
            "-e",
            str(embeddings_path),
            "--batch-size",
            str(attempt_batch_size),
            *_tmbed_gpu_flags(device, cpu_fallback),
            *_tmbed_thread_flags(device, threads),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=_build_tmbed_env(device, model_dir),
        )
        last_result = result
        if result.returncode == 0:
            logger.info(f"[TMbed embed] Saved -> {embeddings_path}")
            return
        if _tmbed_result_looks_resource_killed(result) and attempt_batch_size > 1:
            logger.warning(
                "[TMbed embed] resource kill detected at batch_size=%s; retrying with a smaller batch.",
                attempt_batch_size,
            )
            continue
        break

    stderr = str((last_result.stderr if last_result is not None else "") or "").strip()
    stdout = str((last_result.stdout if last_result is not None else "") or "").strip()
    raise RuntimeError(
        "TMbed embed failed"
        + (
            f" (returncode={last_result.returncode})"
            if last_result is not None
            else ""
        )
        + (f":\n{stderr}" if stderr else (f":\n{stdout}" if stdout else ""))
    )


def _run_tmbed_predict(
    fasta_path: str,
    pred_path: str,
    embeddings_path: Path,
    out_format: int,
    device: str,
    batch_size: int,
    threads: Optional[int],
    model_dir: Path,
    cpu_fallback: bool = True,
) -> None:
    if embeddings_path.exists():
        logger.info(f"[TMbed predict] Reusing embeddings: {embeddings_path}")
    last_result = None
    for attempt_batch_size in _candidate_tmbed_batch_sizes(batch_size):
        logger.info(
            f"[TMbed predict] device={device} out_format={out_format} batch_size={attempt_batch_size}"
        )
        cmd = [
            sys.executable,
            "-m",
            "tmbed",
            "predict",
            "-f",
            fasta_path,
            "-p",
            pred_path,
            "--out-format",
            str(out_format),
            "--batch-size",
            str(attempt_batch_size),
            *_tmbed_gpu_flags(device, cpu_fallback),
            *_tmbed_thread_flags(device, threads),
        ]
        if embeddings_path.exists():
            cmd += ["-e", str(embeddings_path)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=_build_tmbed_env(device, model_dir),
        )
        last_result = result
        if result.returncode == 0:
            return
        if _tmbed_result_looks_resource_killed(result) and attempt_batch_size > 1:
            logger.warning(
                "[TMbed predict] resource kill detected at batch_size=%s; retrying with a smaller batch.",
                attempt_batch_size,
            )
            continue
        break

    stderr = str((last_result.stderr if last_result is not None else "") or "").strip()
    stdout = str((last_result.stdout if last_result is not None else "") or "").strip()
    raise RuntimeError(
        "TMbed predict failed"
        + (
            f" (returncode={last_result.returncode})"
            if last_result is not None
            else ""
        )
        + (f":\n{stderr}" if stderr else (f":\n{stdout}" if stdout else ""))
    )
    pred_file = Path(pred_path)
    if not pred_file.exists() or pred_file.stat().st_size == 0:
        raise RuntimeError(
            f"TMbed produced an empty output file: {pred_path}\n"
            "This usually means the embeddings cache is stale."
        )
    logger.info(f"[TMbed predict] Saved -> {pred_path}")



def _normalize_tm_regions(regions):
    normalized = []
    for index, region in enumerate(regions, start=1):
        try:
            start = int(region["start"])
            end = int(region["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start

        normalized_region = {
            "index": int(region.get("index") or index),
            "start": start,
            "end": end,
            "length": end - start + 1,
        }
        for optional_key in ("label", "chain", "attributes"):
            optional_value = region.get(optional_key)
            if optional_value not in (None, "", {}):
                normalized_region[optional_key] = optional_value
        normalized.append(normalized_region)
    return normalized


def _counts_from_regions(regions_by_sequence):
    return {
        sequence_id: len(regions or [])
        for sequence_id, regions in regions_by_sequence.items()
    }


def extract_tmr_counts_from_gff_text(gff_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in re.finditer(
        r"^#\s*(\S+).*Number of predicted TMRs:\s*(\d+)",
        gff_text,
        re.MULTILINE,
    ):
        sequence_id, count = match.group(1), int(match.group(2))
        counts[sequence_id] = count
    if counts:
        return counts
    for line in gff_text.splitlines():
        parts = line.split("\t")
        if len(parts) > 2 and parts[2] == "TMhelix":
            sequence_id = parts[0]
            counts[sequence_id] = counts.get(sequence_id, 0) + 1
    return counts


def extract_tm_regions_from_gff_text(gff_text: str) -> dict[str, list[dict]]:
    regions: dict[str, list[dict]] = {}
    for line in gff_text.splitlines():
        parts = line.split("\t")
        if len(parts) < 5 or parts[2] != "TMhelix":
            continue
        sequence_id = parts[0]
        attributes = {}
        if len(parts) > 8:
            for item in parts[8].split(";"):
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                attributes[key] = value
        regions.setdefault(sequence_id, []).append(
            {
                "start": parts[3],
                "end": parts[4],
                "label": parts[2],
                "attributes": attributes or None,
            }
        )
    return {
        sequence_id: _normalize_tm_regions(sequence_regions)
        for sequence_id, sequence_regions in regions.items()
    }


_TMBED_LABEL_MAP = {
    "H": ("TM_helix", "IN->OUT"),
    "h": ("TM_helix", "OUT->IN"),
    "B": ("TM_barrel", "IN->OUT"),
    "b": ("TM_barrel", "OUT->IN"),
    "S": ("signal", None),
    ".": ("non_TM", None),
    "i": ("inside", None),
    "o": ("outside", None),
}


def _extract_tmbed_segments(topology: str) -> list[dict]:
    segments = []
    index = 0
    while index < len(topology):
        label = topology[index]
        cursor = index
        while cursor < len(topology) and topology[cursor] == label:
            cursor += 1
        segment_type, orientation = _TMBED_LABEL_MAP.get(label, ("unknown", None))
        segments.append(
            {
                "label": label,
                "type": segment_type,
                "start": index + 1,
                "end": cursor,
                "length": cursor - index,
                "orientation": orientation,
            }
        )
        index = cursor
    return segments


def _extract_tmbed_tm_segments(segments: list[dict]) -> list[dict]:
    return [segment for segment in segments if segment["type"] in {"TM_helix", "TM_barrel"}]


def parse_tmbed_output(pred_path: str, out_format: int = 4) -> dict[str, dict]:
    if out_format not in {0, 1, 4}:
        raise ValueError(f"Unsupported TMbed output format for MetaMP parsing: {out_format}")

    details: dict[str, dict] = {}
    with open(pred_path) as fh:
        lines = [line.rstrip("\n\r") for line in fh if line.strip()]
    if len(lines) % 3 != 0:
        raise ValueError("TMbed 3-line output malformed.")

    for block_start in range(0, len(lines), 3):
        header, sequence, topology = lines[block_start : block_start + 3]
        sequence_id = header.lstrip(">").strip()
        sequence = sequence.strip()
        topology = topology.strip()
        segments = _extract_tmbed_segments(topology)
        tm_segments = _extract_tmbed_tm_segments(segments)
        normalized_regions = _normalize_tm_regions(
            [
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "label": segment["label"],
                    "attributes": {
                        "type": segment["type"],
                        "orientation": segment["orientation"],
                    },
                }
                for segment in tm_segments
            ]
        )
        details[sequence_id] = {
            "sequence": sequence,
            "topology": topology,
            "segments": segments,
            "tm_segments": normalized_regions,
            "count": len(normalized_regions),
            "regions": normalized_regions,
            "has_signal": any(segment["type"] == "signal" for segment in segments),
        }
    return details


def _print_tmbed_result_summary(details: dict[str, dict], limit: int = 5) -> None:
    if not details:
        print("[TMbed] No parsed results to display.")
        return
    for sequence_id, payload in list(details.items())[:limit]:
        topology = str(payload.get("topology") or "")
        print(
            f"[TMbed] {sequence_id}: tm_count={payload.get('count', 0)} "
            f"signal={'yes' if payload.get('has_signal') else 'no'} "
            f"topology={topology[:80]}{'...' if len(topology) > 80 else ''}"
        )
        for index, region in enumerate(payload.get("regions") or [], start=1):
            print(
                f"[TMbed]   region {index}: {region.get('start')}-{region.get('end')} "
                f"label={region.get('label')}"
            )


def _read_fasta_records(fasta_path: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    sequence_lines = []

    with open(fasta_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_lines)))
                header = line[1:].strip()
                sequence_lines = []
            else:
                sequence_lines.append(line)

    if header is not None:
        records.append((header, "".join(sequence_lines)))
    return records


def count_tmbed_3line(pred_path: str) -> dict[str, int]:
    return {
        sequence_id: detail["count"]
        for sequence_id, detail in parse_tmbed_output(pred_path).items()
    }


def serialize_tm_regions(regions):
    if regions is None:
        return ""
    if isinstance(regions, str):
        return regions
    if isinstance(regions, (list, tuple)):
        return json.dumps(list(regions))
    if isinstance(regions, dict):
        return json.dumps(regions)
    if _is_missing_value(regions):
        return ""
    return json.dumps(regions)


def _coerce_tm_regions_json_text(value):
    if _is_missing_value(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        repaired = (
            text.replace("'\"", '"')
            .replace("\"'", '"')
            .replace("''", "'")
        )
        try:
            loaded = json.loads(repaired)
        except json.JSONDecodeError:
            return None

    if not isinstance(loaded, list):
        return None
    try:
        return json.dumps(loaded)
    except (TypeError, ValueError):
        return None


def parse_tm_regions_value(value):
    if _is_missing_value(value):
        return []
    if isinstance(value, list):
        return _normalize_tm_regions(value)
    if isinstance(value, str):
        normalized = _coerce_tm_regions_json_text(value)
        if normalized is None:
            return []
        loaded = json.loads(normalized)
        if isinstance(loaded, list):
            return _normalize_tm_regions(loaded)
    return []


def normalize_tm_regions_json_string(value):
    normalized_regions = parse_tm_regions_value(value)
    return serialize_tm_regions(normalized_regions) if normalized_regions else "[]"


def build_tm_prediction_payload(counts, regions):
    sequence_ids = set((counts or {}).keys()) | set((regions or {}).keys())
    return {
        "counts": counts,
        "regions": {
            sequence_id: _normalize_tm_regions((regions or {}).get(sequence_id) or [])
            for sequence_id in sequence_ids
        },
    }


def extract_tm_prediction_from_text(gff_text: str):
    regions = extract_tm_regions_from_gff_text(gff_text)
    counts = extract_tmr_counts_from_gff_text(gff_text)
    if not counts and regions:
        counts = _counts_from_regions(regions)
    for sequence_id, region_list in regions.items():
        counts.setdefault(sequence_id, len(region_list))
    return build_tm_prediction_payload(counts, regions)


class BasePredictor(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, fasta_path: str) -> dict[str, int]:
        pass

    def predict_with_details(self, fasta_path: str) -> dict[str, dict]:
        counts = self.predict(fasta_path)
        return build_tm_prediction_payload(counts, {})


class TMbedPredictor(BasePredictor):
    def __init__(
        self,
        format_code: int = 4,
        use_gpu: bool = True,
        cpu_fallback: bool = True,
        batch_size: int = 4,
        threads: Optional[int] = None,
    ):
        super().__init__("TMbed")
        self.format_code = format_code
        self.use_gpu = use_gpu
        self.cpu_fallback = cpu_fallback
        self.batch_size = batch_size
        self.threads = threads

    def predict(self, fasta_path: str) -> dict[str, int]:
        return self.predict_with_details(fasta_path)["counts"]

    def predict_with_details(self, fasta_path: str) -> dict[str, dict]:
        _patch_tmbed_utils()
        model_dir = _get_tmbed_model_dir()
        device = detect_tmbed_device(use_gpu=self.use_gpu)
        _ensure_tmbed_model_downloaded(model_dir, device)

        fasta_text = Path(fasta_path).read_text()
        cache_dir = model_dir / "embedding-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = cache_dir / f"{_fasta_hash(fasta_text)}.h5"
        pred_path = fasta_path + ".pred"

        try:
            if not _cache_is_valid(fasta_text, embeddings_path):
                if embeddings_path.exists():
                    embeddings_path.unlink()
                    logger.info(f"[predict_tmbed] Deleted stale cache: {embeddings_path}")
                hash_file = embeddings_path.with_suffix(".hash")
                if hash_file.exists():
                    hash_file.unlink()
                _run_tmbed_embed(
                    fasta_path=fasta_path,
                    embeddings_path=embeddings_path,
                    device=device,
                    batch_size=self.batch_size,
                    threads=self.threads,
                    model_dir=model_dir,
                    cpu_fallback=self.cpu_fallback,
                )
                _write_cache_hash(fasta_text, embeddings_path)

            _run_tmbed_predict(
                fasta_path=fasta_path,
                pred_path=pred_path,
                embeddings_path=embeddings_path,
                out_format=self.format_code,
                device=device,
                batch_size=self.batch_size,
                threads=self.threads,
                model_dir=model_dir,
                cpu_fallback=self.cpu_fallback,
            )
            details = parse_tmbed_output(pred_path, out_format=self.format_code)
            _print_tmbed_result_summary(details)
            return build_tm_prediction_payload(
                {sequence_id: payload["count"] for sequence_id, payload in details.items()},
                {sequence_id: payload["regions"] for sequence_id, payload in details.items()},
            )
        except Exception as exc:
            print(f"[TMbed] Batch prediction failed: {exc}")
        finally:
            if os.path.exists(pred_path):
                os.remove(pred_path)

        records = _read_fasta_records(fasta_path)
        if len(records) <= 1:
            return build_tm_prediction_payload({}, {})

        print(
            f"[TMbed] Falling back to per-sequence recovery for {len(records)} sequences."
        )
        recovered_details: dict[str, dict] = {}
        failed_ids: list[str] = []

        for sequence_id, sequence in records:
            try:
                with NamedTemporaryFile("w+", suffix=".fasta", delete=True) as single_fasta:
                    single_fasta.write(f">{sequence_id}\n{sequence}\n")
                    single_fasta.flush()
                    single_pred_path = single_fasta.name + ".pred"
                    single_fasta_text = Path(single_fasta.name).read_text()
                    single_embeddings_path = (
                        model_dir
                        / "embedding-cache"
                        / f"{_fasta_hash(single_fasta_text)}.h5"
                    )
                    if not _cache_is_valid(single_fasta_text, single_embeddings_path):
                        if single_embeddings_path.exists():
                            single_embeddings_path.unlink()
                        hash_file = single_embeddings_path.with_suffix(".hash")
                        if hash_file.exists():
                            hash_file.unlink()
                        _run_tmbed_embed(
                            fasta_path=single_fasta.name,
                            embeddings_path=single_embeddings_path,
                            device=device,
                            batch_size=self.batch_size,
                            threads=self.threads,
                            model_dir=model_dir,
                            cpu_fallback=self.cpu_fallback,
                        )
                        _write_cache_hash(single_fasta_text, single_embeddings_path)

                    _run_tmbed_predict(
                        fasta_path=single_fasta.name,
                        pred_path=single_pred_path,
                        embeddings_path=single_embeddings_path,
                        out_format=self.format_code,
                        device=device,
                        batch_size=self.batch_size,
                        threads=self.threads,
                        model_dir=model_dir,
                        cpu_fallback=self.cpu_fallback,
                    )
                    parsed_single = parse_tmbed_output(single_pred_path, out_format=self.format_code)
                    _print_tmbed_result_summary(parsed_single, limit=1)
                    recovered_details.update(parsed_single)
                    if os.path.exists(single_pred_path):
                        os.remove(single_pred_path)
            except Exception:
                traceback.print_exc()
                failed_ids.append(sequence_id)

        if failed_ids:
            print(
                f"[TMbed] Unable to recover predictions for {len(failed_ids)} sequence(s): "
                + ", ".join(failed_ids[:10])
                + (" ..." if len(failed_ids) > 10 else "")
            )

        details = recovered_details
        return build_tm_prediction_payload(
            {sequence_id: payload["count"] for sequence_id, payload in details.items()},
            {sequence_id: payload["regions"] for sequence_id, payload in details.items()},
        )


class DeepTMHMMPredictor(BasePredictor):
    def __init__(self, spec: str = "DTU/DeepTMHMM:1.0.24"):
        super().__init__("DeepTMHMM")
        try:
            import biolib
        except ImportError:
            raise RuntimeError("Install biolib for DeepTMHMMPredictor")
        self.app = biolib.load(spec)

    def predict(self, fasta_path: str) -> dict[str, int]:
        return self.predict_with_details(fasta_path)["counts"]

    def predict_with_details(self, fasta_path: str) -> dict[str, dict]:
        job = self.app.run(fasta=fasta_path)

        print(f"[DeepTMHMM] Running job {job.id}, waiting for output...")

        # This blocks until the file is ready or the job fails internally
        fh = job.get_output_file('/deeptmhmm_results.md').get_file_handle()
        fh.seek(0)
        txt = fh.read().decode()

        print(f"[DeepTMHMM] Job {job.id} completed successfully.")

        return extract_tm_prediction_from_text(txt)


class MultiModelAnalyzer:
    def __init__(self,
                 db_params: dict,
                 table: str,
                 batch_size: int = 10,
                 max_workers: int = 1,
                 max_sequences: int = 4,
                 use_db: bool = True,
                 write_csv: bool = False,
                 force_predictors: Optional[set[str]] = None):
        self.predictors: list[BasePredictor] = []
        self.db_params = db_params
        self.table = table
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_sequences = max_sequences
        self.use_db = use_db
        self.write_csv = write_csv
        self.force_predictors = {str(name) for name in (force_predictors or set())}

    def register(self, predictor):
        self.predictors.append(predictor)

    @staticmethod
    def _completed_predictor_mask(dataframe: pd.DataFrame, predictor_name: str) -> pd.Series:
        count_col = f"{predictor_name}_tm_count"
        region_col = f"{predictor_name}_tm_regions"

        has_any_completion_column = False
        completion_mask = pd.Series(True, index=dataframe.index)

        if count_col in dataframe.columns:
            has_any_completion_column = True
            completion_mask &= dataframe[count_col].apply(_has_meaningful_value)
        if region_col in dataframe.columns:
            has_any_completion_column = True
            completion_mask &= dataframe[region_col].apply(_has_meaningful_value)

        if has_any_completion_column:
            return completion_mask
        return pd.Series(False, index=dataframe.index)

    def analyze(self,
        df: pd.DataFrame,
        id_col: str,
        seq_col: str,
        csv_out: Optional[str] = None,
        resume_from_csv: Optional[str] = None,
        on_batch_completed=None) -> pd.DataFrame:
        df = df.reset_index(drop=True)

        if resume_from_csv and os.path.exists(resume_from_csv):
            done_df = pd.read_csv(resume_from_csv)
            if id_col in done_df.columns:
                completion_columns = [
                    f"{predictor.name}_tm_count"
                    for predictor in self.predictors
                    if f"{predictor.name}_tm_count" in done_df.columns
                ]
                if completion_columns:
                    complete_mask = done_df[completion_columns].apply(
                        lambda column: column.map(_has_meaningful_value)
                    ).all(axis=1)
                    done_df = done_df.loc[complete_mask]
                done_ids = {
                    str(value).strip()
                    for value in done_df[id_col].dropna().tolist()
                    if str(value).strip()
                }
                before_filter = len(df)
                df = df[
                    ~df[id_col].astype(str).str.strip().isin(done_ids)
                ]
                skipped = before_filter - len(df)
                print(
                    f"[RESUME] Skipping {skipped} already processed entries."
                )
            else:
                done_ids = set()
        else:
            done_df = pd.DataFrame()

        total = len(df)
        if total == 0:
            print("All sequences are already processed.")
            return done_df

        print(f"🚀 Starting analysis of {total} sequences in batches of {self.batch_size}...")

        for p in self.predictors:
            count_col = f"{p.name}_tm_count"
            region_col = f"{p.name}_tm_regions"
            if count_col not in df.columns:
                df[count_col] = pd.NA
            if region_col not in df.columns:
                df[region_col] = ""

        if self.use_db:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            cur.execute(sql.SQL(
                "ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {seq_col} TEXT"
            ).format(tbl=sql.Identifier(self.table), seq_col=sql.Identifier(seq_col)))

            for p in self.predictors:
                cur.execute(sql.SQL(
                    "ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} INTEGER"
                ).format(tbl=sql.Identifier(self.table), col=sql.Identifier(f"{p.name}_tm_count")))
                cur.execute(sql.SQL(
                    "ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} TEXT"
                ).format(tbl=sql.Identifier(self.table), col=sql.Identifier(f"{p.name}_tm_regions")))

            conn.commit()
        else:
            conn = cur = None

        results = []

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = df.iloc[start:end]
            print(f"\n🔄 Batch {start}-{end}...")

            try:
                # 1. Store sequences
                if self.use_db:
                    update_seqs = [(row[seq_col], row[id_col]) for _, row in batch.iterrows()]
                    cur.executemany(
                        sql.SQL("UPDATE {tbl} SET {seq_col} = %s WHERE {id_col} = %s").format(
                            tbl=sql.Identifier(self.table),
                            seq_col=sql.Identifier(seq_col),
                            id_col=sql.Identifier(id_col)
                        ).as_string(conn),
                        update_seqs
                    )
                    conn.commit()
                    print("Sequences committed to DB")

                # 2. Write FASTA
                with NamedTemporaryFile("w+", suffix=".fasta", delete=True) as fasta:
                    for _, row in batch.iterrows():
                        fasta.write(f">{row[id_col]}\n{row[seq_col]}\n")
                    fasta.flush()

                    def _run(p):
                        try:
                            predictor_batch = (
                                batch.copy()
                                if p.name in self.force_predictors
                                else batch.loc[
                                    ~self._completed_predictor_mask(batch, p.name)
                                ].copy()
                            )
                            if predictor_batch.empty:
                                print(f"⏭️ Skipping predictor {p.name}; this batch already has values.")
                                return p.name, build_tm_prediction_payload({}, {})

                            with NamedTemporaryFile("w+", suffix=".fasta", delete=True) as predictor_fasta:
                                for _, row in predictor_batch.iterrows():
                                    predictor_fasta.write(f">{row[id_col]}\n{row[seq_col]}\n")
                                predictor_fasta.flush()

                                print(
                                    f"🔍 Running predictor: {p.name} on {len(predictor_batch)} sequence(s)"
                                )
                                result = p.predict_with_details(predictor_fasta.name)
                            predicted_sequences = len(result.get("counts", {}) or {})
                            if predicted_sequences:
                                print(
                                    f"{p.name} completed with predictions for {predicted_sequences} sequence(s)."
                                )
                            else:
                                print(
                                    f"{p.name} completed without usable predictions for this batch."
                                )
                            return p.name, result
                        except Exception as e:
                            traceback.print_exc()
                            print(f"{p.name} failed: {e}")
                            return p.name, build_tm_prediction_payload({}, {})

                    results_map = (
                        list(ThreadPoolExecutor(max_workers=self.max_workers).map(_run, self.predictors))
                        if self.max_workers > 1 else
                        [_run(p) for p in self.predictors]
                    )

                    for name, prediction_result in results_map:
                        counts = prediction_result.get("counts", {})
                        regions = prediction_result.get("regions", {})
                        count_col = f"{name}_tm_count"
                        region_col = f"{name}_tm_regions"
                        for sid, ct in counts.items():
                            batch.loc[batch[id_col] == sid, count_col] = int(ct)
                        for sid, tm_regions in regions.items():
                            batch.loc[batch[id_col] == sid, region_col] = serialize_tm_regions(tm_regions)

                # 3. Store predictions
                if self.use_db:
                    updates = []
                    for _, row in batch.iterrows():
                        values = []
                        for p in self.predictors:
                            values.append(_safe_optional_int(row[f"{p.name}_tm_count"]))
                            values.append(row[f"{p.name}_tm_regions"])
                        values.append(row[id_col])
                        updates.append(tuple(values))

                    set_clause = sql.SQL(", ").join(
                        sql.SQL("{} = %s, {} = %s").format(
                            sql.Identifier(f"{p.name}_tm_count"),
                            sql.Identifier(f"{p.name}_tm_regions"),
                        )
                        for p in self.predictors
                    )
                    update_query = sql.SQL("UPDATE {tbl} SET {sets} WHERE {id_col} = %s").format(
                        tbl=sql.Identifier(self.table),
                        sets=set_clause,
                        id_col=sql.Identifier(id_col)
                    )
                    cur.executemany(update_query.as_string(conn), updates)
                    conn.commit()
                    print("📥 TM count updates committed to DB")

                results.append(batch)

                # 4. Save interim results
                combined = pd.concat([done_df] + results, ignore_index=True)
                if self.write_csv and csv_out:
                    combined.to_csv(csv_out, index=False)
                    print(f"💾 Batch {start}-{end} progress saved to {csv_out}")
                if on_batch_completed is not None:
                    on_batch_completed(
                        batch.copy(),
                        start=start,
                        end=end,
                        total=total,
                    )

            except Exception as e:
                traceback.print_exc()
                print(f"❌ Batch {start}-{end} failed: {e}")
                continue

        if self.use_db:
            cur.close()
            conn.close()

        final_df = pd.concat([done_df] + results, ignore_index=True)
        if self.write_csv and csv_out:
            final_df.to_csv(csv_out, index=False)

        print("Finished all batches.")
        return final_df

    

# class MultiModelAnalyzer:
#     """
#     Runs predictors on a DataFrame, updates Postgres in batches, and writes CSV.
#     """
#     def __init__(self,
#                  db_params: dict,
#                  table: str,
#                  batch_size: int = 10,
#                  max_workers: int = 1):
#         self.predictors: list[BasePredictor] = []
#         self.db_params = db_params
#         self.table = table
#         self.batch_size = batch_size
#         self.max_workers = max_workers

#     def register(self, p: BasePredictor):
#         self.predictors.append(p)

#     def analyze(self,
#                 df: pd.DataFrame,
#                 id_col: str,
#                 seq_col: str,
#                 csv_out: str) -> pd.DataFrame:

#         # 1) Reset index so .loc works predictably
#         df = df.reset_index(drop=True)
#         total = len(df)

#         # 2) Add any missing TM count columns
#         for p in self.predictors:
#             df[f"{p.name}_tm_count"] = pd.NA

#         # 3) Open one DB connection & create missing columns
#         conn = psycopg2.connect(**self.db_params)
#         cur = conn.cursor()

#         # sequence_sequence column (TEXT)
#         cur.execute(sql.SQL(
#             "ALTER TABLE {tbl} "
#             "ADD COLUMN IF NOT EXISTS {seq_col} TEXT"
#         ).format(
#             tbl=sql.Identifier(self.table),
#             seq_col=sql.Identifier(seq_col),
#         ))
        
#         for p in self.predictors:
#             col = f"{p.name}_tm_count"
#             cur.execute(sql.SQL(
#                 "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} INTEGER"
#             ).format(
#                 table=sql.Identifier(self.table),
#                 col=sql.Identifier(col),
#             ))
#         conn.commit()

#         # 4) Process in fixed-size batches
#         batch_num = 0
#         for start in range(0, total, self.batch_size):
#             batch_num += 1
#             end = min(start + self.batch_size, total)
#             batch = df.iloc[start:end]

#             # 4a) Write FASTA to a NamedTemporaryFile
#             with NamedTemporaryFile("w+", suffix=".fasta", delete=True) as fasta:
#                 for _, row in batch.iterrows():
#                     fasta.write(f">{row[id_col]}\n{row[seq_col]}\n")
#                 fasta.flush()

#                 # 4b) Run predictors (optionally in parallel)
#                 def _run(predictor):
#                     return predictor.name, predictor.predict(fasta.name)

#                 if self.max_workers > 1:
#                     with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
#                         results = list(ex.map(_run, self.predictors))
#                 else:
#                     results = [_run(p) for p in self.predictors]

#             # 4c) Populate counts back into df
#             for name, counts in results:
#                 col = f"{name}_tm_count"
#                 for sid, ct in counts.items():
#                     df.loc[df[id_col] == sid, col] = int(ct)

#             # 4d) Bulk-update Postgres for this batch
#             updates = []
#             for _, row in batch.iterrows():
#                 vals = [int(row[f"{p.name}_tm_count"]) for p in self.predictors]
#                 vals.append(row[id_col])
#                 updates.append(tuple(vals))

#             cols = [sql.Identifier(seq_col)] + [sql.Identifier(f"{p.name}_tm_count") for p in self.predictors]
#             set_clause = sql.SQL(", ").join(
#                 sql.SQL("{} = %s").format(c) for c in cols
#             )
#             query = sql.SQL(
#                 "UPDATE {table} SET {sets} WHERE {id_col} = %s"
#             ).format(
#                 table=sql.Identifier(self.table),
#                 sets=set_clause,
#                 id_col=sql.Identifier(id_col)
#             )
#             cur.executemany(query.as_string(conn), updates)
#             conn.commit()

#             print(f"Processed batch #{batch_num}: rows {start+1}-{end}")

#         # 5) Cleanup DB connection
#         cur.close()
#         conn.close()

#         # 6) Write out the full DataFrame (with TM counts) to CSV
#         df.to_csv(csv_out, index=False)
#         print(f"Saved CSV: {csv_out} ({len(df)} rows)")
#         return df
