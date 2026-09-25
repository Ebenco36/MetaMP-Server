#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_COLUMNS = [
    "id",
    "pdb_code",
    "uniprot_id",
    "provider",
    "method",
    "prediction_kind",
    "tm_count",
    "tm_regions_json",
    "raw_payload_json",
    "source_url",
    "status",
    "error_message",
    "created_at",
    "updated_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean a malformed TMAlphaFold export CSV and upsert it into "
            "membrane_protein_tmalphafold_predictions by id."
        )
    )
    parser.add_argument("export_path", help="Path to the exported CSV file.")
    parser.add_argument(
        "--db-container",
        default=os.getenv("TMALPHA_DB_CONTAINER", "testmetaMPDB"),
        help="Docker container name for Postgres. Default: testmetaMPDB",
    )
    parser.add_argument(
        "--db-user",
        default=os.getenv("TMALPHA_DB_USER", "mpvis_user"),
        help="Database user. Default: mpvis_user",
    )
    parser.add_argument(
        "--db-name",
        default=os.getenv("TMALPHA_DB_NAME", "mpvis_db"),
        help="Database name. Default: mpvis_db",
    )
    parser.add_argument(
        "--cleaned-output",
        default=None,
        help=(
            "Optional path for the cleaned CSV. Default: sibling file ending in "
            "'.fixed.csv'."
        ),
    )
    parser.add_argument(
        "--keep-container-copy",
        action="store_true",
        help="Keep the temporary cleaned CSV inside the Postgres container.",
    )
    return parser.parse_args()


def parse_loose_csv_records(text: str) -> list[list[str]]:
    records: list[list[str]] = []
    fields: list[str] = []
    buf: list[str] = []
    in_quoted = False
    at_field_start = True
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_quoted:
            if ch == '"' and (nxt == "," or nxt == "\n" or nxt == "\r" or nxt == ""):
                in_quoted = False
            else:
                buf.append(ch)
            i += 1
            at_field_start = False
            continue

        if at_field_start and ch == '"':
            in_quoted = True
            i += 1
            at_field_start = False
            continue

        if ch == ",":
            fields.append("".join(buf))
            buf = []
            at_field_start = True
            i += 1
            continue

        if ch == "\n":
            fields.append("".join(buf))
            if any(field.strip() for field in fields):
                records.append(fields)
            fields = []
            buf = []
            at_field_start = True
            i += 1
            continue

        if ch == "\r":
            i += 1
            continue

        buf.append(ch)
        at_field_start = False
        i += 1

    if buf or fields:
        fields.append("".join(buf))
        if any(field.strip() for field in fields):
            records.append(fields)

    return records


def normalize_row(fields: list[str]) -> list[str]:
    if len(fields) == 14:
        return fields

    if len(fields) < 14:
        raise ValueError(f"Row has too few fields: {len(fields)}")

    head = fields[:7]
    tail = fields[-5:]
    middle = fields[7:-5]
    if len(middle) < 2:
        raise ValueError(f"Row middle section is too short: {len(middle)}")

    tm_regions_json = middle[0]
    raw_payload_json = ",".join(middle[1:])
    normalized = head + [tm_regions_json, raw_payload_json] + tail
    if len(normalized) != 14:
        raise ValueError(f"Normalized row has wrong size: {len(normalized)}")
    return normalized


def clean_export_file(src: Path, dest: Path) -> tuple[int, int]:
    text = src.read_text(encoding="utf-8", errors="replace")
    records = parse_loose_csv_records(text)
    repaired_rows = 0

    with dest.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(EXPECTED_COLUMNS)

        for parsed in records:
            normalized = normalize_row(parsed)
            if len(parsed) != 14:
                repaired_rows += 1
            writer.writerow(normalized)

    return len(records), repaired_rows


def run_checked(cmd: list[str], *, input_text: str | None = None) -> None:
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True if input_text is not None else None,
        capture_output=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout or "")
        raise SystemExit(result.returncode)


def copy_file_into_container(local_path: Path, container: str, container_path: str) -> None:
    run_checked(["docker", "cp", str(local_path), f"{container}:{container_path}"])


def remove_file_in_container(container: str, container_path: str) -> None:
    run_checked(["docker", "exec", container, "rm", "-f", container_path])


def import_cleaned_csv(
    *,
    container: str,
    db_user: str,
    db_name: str,
    container_csv_path: str,
) -> None:
    columns_sql = ", ".join(EXPECTED_COLUMNS)
    updates_sql = ", ".join(
        f"{column} = source.{column}" for column in EXPECTED_COLUMNS if column != "id"
    )
    natural_key_match_sql = (
        "target.provider = source.provider "
        "AND target.method = source.method "
        "AND target.pdb_code = source.pdb_code "
        "AND target.uniprot_id = source.uniprot_id"
    )
    sequence_sql = (
        "SELECT setval("
        "pg_get_serial_sequence('membrane_protein_tmalphafold_predictions','id'), "
        "COALESCE(MAX(id), 1), true"
        ") "
        "FROM membrane_protein_tmalphafold_predictions;"
    )

    sql_script = f"""
BEGIN;
CREATE TEMP TABLE tmalpha_import_staging (LIKE membrane_protein_tmalphafold_predictions INCLUDING DEFAULTS);
\\copy tmalpha_import_staging ({columns_sql}) FROM '{container_csv_path}' WITH (FORMAT csv, HEADER true);
CREATE TEMP TABLE tmalpha_import_ranked AS
WITH ranked AS (
    SELECT
        {columns_sql},
        ROW_NUMBER() OVER (
            PARTITION BY id
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        ) AS rn
    FROM tmalpha_import_staging
)
SELECT {columns_sql}
FROM ranked
WHERE rn = 1;

DELETE FROM membrane_protein_tmalphafold_predictions AS target
USING tmalpha_import_ranked AS source
WHERE target.id = source.id
  AND NOT ({natural_key_match_sql});

UPDATE membrane_protein_tmalphafold_predictions AS target
SET
    id = source.id,
    {updates_sql}
FROM tmalpha_import_ranked AS source
WHERE {natural_key_match_sql};

INSERT INTO membrane_protein_tmalphafold_predictions ({columns_sql})
SELECT {columns_sql}
FROM tmalpha_import_ranked AS source
WHERE NOT EXISTS (
    SELECT 1
    FROM membrane_protein_tmalphafold_predictions AS target
    WHERE target.id = source.id
       OR ({natural_key_match_sql})
);

{sequence_sql}
COMMIT;
"""

    run_checked(
        [
            "docker",
            "exec",
            "-i",
            container,
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            db_user,
            "-d",
            db_name,
        ],
        input_text=sql_script,
    )


def resolve_cleaned_output_path(src: Path, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()
    return src.with_name(f"{src.stem}.fixed{src.suffix}")


def main() -> int:
    args = parse_args()
    src = Path(args.export_path).expanduser().resolve()
    if not src.exists():
        print(f"Export file not found: {src}", file=sys.stderr)
        return 2

    if shutil.which("docker") is None:
        print("Missing required command: docker", file=sys.stderr)
        return 2

    cleaned_output = resolve_cleaned_output_path(src, args.cleaned_output)
    cleaned_output.parent.mkdir(parents=True, exist_ok=True)

    rows, repaired_rows = clean_export_file(src, cleaned_output)
    print(
        f"Cleaned {rows} row(s) from {src} into {cleaned_output} "
        f"({repaired_rows} repaired row(s))."
    )

    container_csv_path = f"/tmp/{cleaned_output.name}"
    copy_file_into_container(cleaned_output, args.db_container, container_csv_path)
    try:
        import_cleaned_csv(
            container=args.db_container,
            db_user=args.db_user,
            db_name=args.db_name,
            container_csv_path=container_csv_path,
        )
    finally:
        if not args.keep_container_copy:
            remove_file_in_container(args.db_container, container_csv_path)

    print(
        "Upsert completed into membrane_protein_tmalphafold_predictions "
        f"using database {args.db_name} in container {args.db_container}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
