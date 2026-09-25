#!/usr/bin/env sh
set -eu

INPUT_FASTA=""
OUTPUT_CSV=""
REFERENCE_CSV=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --input)
      INPUT_FASTA="$2"
      shift 2
      ;;
    --output)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --reference)
      REFERENCE_CSV="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$INPUT_FASTA" ] || [ -z "$OUTPUT_CSV" ]; then
  echo "Usage: $0 --input <fasta> --output <csv> [--reference <csv>]" >&2
  exit 2
fi

cat > "$OUTPUT_CSV" <<'EOF'
pdb_code,tm_count,tm_regions
EOF

echo "Wrapper template wrote an empty normalized results CSV to $OUTPUT_CSV" >&2
if [ -n "$REFERENCE_CSV" ] && [ -f "$REFERENCE_CSV" ]; then
  echo "Reference CSV is available at $REFERENCE_CSV" >&2
fi
