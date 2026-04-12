#!/usr/bin/env python
import argparse
import json
import os
import sys
import statistics
import platform
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server import app
from src.AI_Packages.TMProteinPredictor import parse_tm_regions_value
from src.MP.model_mpstruct import MPSTURC
from src.MP.model_tmalphafold import TMAlphaFoldPrediction


def _is_pathological_topology(row) -> bool:
    regions = parse_tm_regions_value(row.tm_regions_json)
    raw_payload = row.raw_payload_json or ""
    topology = ""
    try:
        payload = json.loads(raw_payload) if raw_payload else {}
        topology = str(payload.get("topology") or "")
    except Exception:
        topology = ""

    if topology:
        unique_labels = {label for label in topology if label.strip()}
        if len(unique_labels) == 1 and unique_labels.issubset({"B", "b"}):
            return True

    if len(regions) != 1:
        return False

    region = regions[0] or {}
    label = str(region.get("label") or "").strip()
    region_length = int(region.get("length") or 0)

    seq_length = 0
    if topology:
        seq_length = len(topology)
    elif raw_payload:
        try:
            payload = json.loads(raw_payload)
            seq_length = len(str(payload.get("sequence") or ""))
        except Exception:
            seq_length = 0

    if seq_length <= 0:
        seq_length = region_length

    coverage = region_length / max(1, seq_length)
    if label in {"B", "b"} and coverage >= 0.90:
        return True
    if label in {"H", "h"} and coverage >= 0.95:
        return True
    return False


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate TMbed predictions against basic quality gates and peer predictors."
    )
    parser.add_argument("--provider", default="MetaMP")
    parser.add_argument("--method", default="TMbed")
    parser.add_argument(
        "--comparators",
        default="DeepTMHMM,TMHMM,TMDET",
        help="Comma-separated comparator methods.",
    )
    parser.add_argument("--one-tm-threshold", type=float, default=0.20)
    parser.add_argument("--pathological-threshold", type=float, default=0.00)
    parser.add_argument("--strong-mismatch-threshold", type=float, default=0.10)
    parser.add_argument("--sample-limit", type=int, default=12)
    parser.add_argument("--allow-empty", action="store_true")
    return parser


def main() -> int:
    args = _build_argument_parser().parse_args()
    comparator_methods = [item.strip() for item in args.comparators.split(",") if item.strip()]

    with app.app_context():
        target_rows = (
            TMAlphaFoldPrediction.query.filter_by(
                provider=args.provider,
                method=args.method,
                status="success",
            )
            .all()
        )

        if not target_rows:
            machine = str(platform.machine() or "").strip().lower()
            allow_arm64 = str(os.getenv("TMBED_ALLOW_ARM64", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if args.allow_empty or (machine in {"arm64", "aarch64"} and not allow_arm64):
                print("TMbed Quality Report")
                print("rows=0")
                print("No successful TMbed rows found.")
                print("PASS")
                return 0
            print("No successful TMbed rows found for validation.")
            return 1

        comparator_rows = (
            TMAlphaFoldPrediction.query.filter(
                TMAlphaFoldPrediction.method.in_(comparator_methods),
                TMAlphaFoldPrediction.status == "success",
            )
            .all()
        )

        group_map = {
            record.pdb_code: record.group
            for record in MPSTURC.query.filter(
                MPSTURC.pdb_code.in_({row.pdb_code for row in target_rows if row.pdb_code})
            ).all()
        }

        tm_count_distribution = Counter()
        pathological_rows = []
        one_tm_rows = []

        for row in target_rows:
            tm_count_distribution[int(row.tm_count or 0)] += 1
            if int(row.tm_count or 0) == 1:
                one_tm_rows.append(row)
            if _is_pathological_topology(row):
                pathological_rows.append(row)

        comparator_map = defaultdict(list)
        for row in comparator_rows:
            comparator_map[row.pdb_code].append(
                {
                    "provider": row.provider,
                    "method": row.method,
                    "tm_count": int(row.tm_count or 0),
                }
            )

        strong_mismatches = []
        for row in target_rows:
            comparators = comparator_map.get(row.pdb_code) or []
            comparator_counts = [item["tm_count"] for item in comparators if item["tm_count"] is not None]
            if not comparator_counts:
                continue
            median_count = statistics.median(comparator_counts)
            if int(row.tm_count or 0) == 1 and median_count >= 2:
                strong_mismatches.append(
                    {
                        "pdb_code": row.pdb_code,
                        "uniprot_id": row.uniprot_id,
                        "group": group_map.get(row.pdb_code),
                        "tmbed_tm_count": int(row.tm_count or 0),
                        "comparator_tm_counts": comparators,
                    }
                )

        total = len(target_rows)
        one_tm_rate = len(one_tm_rows) / max(1, total)
        pathological_rate = len(pathological_rows) / max(1, total)
        strong_mismatch_rate = len(strong_mismatches) / max(1, total)

        print("TMbed Quality Report")
        print(f"rows={total}")
        print(f"one_tm_rows={len(one_tm_rows)} rate={one_tm_rate:.3f}")
        print(f"pathological_rows={len(pathological_rows)} rate={pathological_rate:.3f}")
        print(f"strong_mismatch_rows={len(strong_mismatches)} rate={strong_mismatch_rate:.3f}")
        print(f"tm_count_distribution={dict(tm_count_distribution.most_common(12))}")

        if pathological_rows:
            print("\nSample pathological rows:")
            for row in pathological_rows[: args.sample_limit]:
                print(
                    f"- {row.pdb_code} {row.uniprot_id} tm_count={row.tm_count} "
                    f"group={group_map.get(row.pdb_code)}"
                )

        if strong_mismatches:
            print("\nSample TMbed vs comparator mismatches:")
            for item in strong_mismatches[: args.sample_limit]:
                comparator_summary = ", ".join(
                    f"{entry['provider']}/{entry['method']}={entry['tm_count']}"
                    for entry in item["comparator_tm_counts"]
                )
                print(
                    f"- {item['pdb_code']} {item['uniprot_id']} "
                    f"TMbed={item['tmbed_tm_count']} | {comparator_summary}"
                )

        failures = []
        if one_tm_rate > args.one_tm_threshold:
            failures.append(
                f"one_tm_rate {one_tm_rate:.3f} exceeds threshold {args.one_tm_threshold:.3f}"
            )
        if pathological_rate > args.pathological_threshold:
            failures.append(
                f"pathological_rate {pathological_rate:.3f} exceeds threshold {args.pathological_threshold:.3f}"
            )
        if strong_mismatch_rate > args.strong_mismatch_threshold:
            failures.append(
                "strong_mismatch_rate "
                f"{strong_mismatch_rate:.3f} exceeds threshold {args.strong_mismatch_threshold:.3f}"
            )

        if failures:
            print("\nFAIL")
            for item in failures:
                print(f"- {item}")
            return 1

        print("\nPASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
