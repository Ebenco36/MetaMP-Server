#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server import app
from src.AI_Packages.TMProteinPredictor import (
    _get_tmbed_model_dir,
    get_tmbed_runtime_status,
    parse_tm_regions_value,
    parse_tmbed_output,
)
from src.MP.model import MembraneProteinData
from src.MP.model_tmalphafold import TMAlphaFoldPrediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TMbed for one or more PDB codes and compare with stored rows."
    )
    parser.add_argument("pdb_codes", nargs="+", help="One or more PDB codes to probe.")
    parser.add_argument("--provider", default="MetaMP")
    parser.add_argument("--method", default="TMbed")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def _summarize_regions(value) -> str:
    regions = parse_tm_regions_value(value)
    if not regions:
        return "[]"
    preview = []
    for region in regions[:4]:
        preview.append(
            f"{region.get('label')}:{region.get('start')}-{region.get('end')}"
        )
    if len(regions) > 4:
        preview.append("...")
    return "[" + ", ".join(preview) + "]"


def _print_stored_rows(pdb_codes: list[str], provider: str, method: str) -> list[str]:
    available = []
    print("=== Stored rows ===")
    with app.app_context():
        for pdb_code in pdb_codes:
            rows = (
                TMAlphaFoldPrediction.query.filter_by(
                    provider=provider,
                    method=method,
                    pdb_code=pdb_code,
                )
                .order_by(TMAlphaFoldPrediction.id.asc())
                .all()
            )
            record = MembraneProteinData.query.filter_by(pdb_code=pdb_code).first()
            sequence = (getattr(record, "sequence_sequence", "") or "").strip()
            if sequence:
                available.append(pdb_code)
            print(
                f"{pdb_code}: sequence_length={len(sequence)} "
                f"stored_rows={len(rows)}"
            )
            for row in rows:
                print(
                    f"  id={row.id} tm_count={row.tm_count} status={row.status} "
                    f"regions={_summarize_regions(row.tm_regions_json)}"
                )
            if not rows:
                print("  no stored rows")
    return available


def _write_fasta(temp_dir: Path, pdb_codes: list[str]) -> Path:
    fasta_path = temp_dir / "tmbed_probe.fa"
    fasta_chunks: list[str] = []
    missing: list[str] = []
    with app.app_context():
        for pdb_code in pdb_codes:
            record = MembraneProteinData.query.filter_by(pdb_code=pdb_code).first()
            sequence = (getattr(record, "sequence_sequence", "") or "").strip()
            if not sequence:
                missing.append(pdb_code)
                continue
            fasta_chunks.append(f">{pdb_code}\n{sequence}\n")
    if missing:
        print(f"Missing sequence(s): {', '.join(missing)}")
    fasta_path.write_text("".join(fasta_chunks))
    return fasta_path


def _run_tmbed(fasta_path: Path, output_path: Path, batch_size: int) -> subprocess.CompletedProcess:
    model_dir = _get_tmbed_model_dir()
    cmd = [
        "/opt/venv/bin/python",
        "-m",
        "tmbed",
        "predict",
        "--fasta",
        str(fasta_path),
        "--predictions",
        str(output_path),
        "--out-format",
        "0",
        "--batch-size",
        str(batch_size),
        "--no-use-gpu",
        "--cpu-fallback",
    ]
    cmd.extend(["--model-dir", str(model_dir)])
    env = os.environ.copy()
    env.setdefault("TMBED_MODEL_DIR", str(model_dir))
    env.setdefault("HF_HOME", str(model_dir / "hf-home"))
    env.setdefault("XDG_CACHE_HOME", str(model_dir / ".cache"))
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def main() -> int:
    args = build_parser().parse_args()
    pdb_codes = [code.strip().upper() for code in args.pdb_codes if code.strip()]
    if not pdb_codes:
        print("No PDB codes provided.")
        return 1

    available_codes = _print_stored_rows(pdb_codes, args.provider, args.method)
    if not available_codes:
        print("No sequences available to probe.")
        return 1

    status = get_tmbed_runtime_status(use_gpu=False)
    print("=== Runtime status ===")
    for key in [
        "available",
        "reason",
        "machine",
        "device",
        "transformers_version",
        "tmbed_version",
        "cnn_model_count",
        "smoke_test_tm_count",
    ]:
        if key in status:
            print(f"{key}={status[key]}")
    if not status.get("available"):
        return 1

    temp_dir_ctx = tempfile.TemporaryDirectory(prefix="tmbed_probe_")
    temp_dir = Path(temp_dir_ctx.name)
    fasta_path = _write_fasta(temp_dir, available_codes)
    output_path = temp_dir / "tmbed_probe.out"

    print("=== Running TMbed ===")
    result = _run_tmbed(fasta_path, output_path, args.batch_size)

    if result.stdout.strip():
        print("\n=== TMbed stdout ===")
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print("\n=== TMbed stderr ===")
        print(result.stderr.rstrip())

    print(f"\nreturncode={result.returncode}")

    if not output_path.exists():
        print(f"No TMbed output file was produced at {output_path}.")
        if not args.keep_temp:
            temp_dir_ctx.cleanup()
        return result.returncode or 1

    print("\n=== Raw TMbed output ===")
    print(output_path.read_text().rstrip())

    print("\n=== Parsed summary ===")
    try:
        parsed = parse_tmbed_output(output_path, out_format=0)
    except Exception as exc:
        print(f"Parser rejected TMbed output: {exc}")
        if args.keep_temp:
            print(f"Temporary files kept in {temp_dir}")
        else:
            temp_dir_ctx.cleanup()
        return 1

    for sequence_id, payload in parsed.items():
        regions = payload.get("regions") or []
        summary = ", ".join(
            f"{region.get('label')}:{region.get('start')}-{region.get('end')}"
            for region in regions
        )
        print(
            f"{sequence_id}: tm_count={payload.get('count')} "
            f"regions=[{summary}]"
        )

    if args.keep_temp:
        print(f"\nTemporary files kept in {temp_dir}")
    else:
        temp_dir_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
