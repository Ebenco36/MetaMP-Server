from __future__ import annotations

import re


CANONICAL_GROUP_LABELS = {
    "monotopic membrane proteins": "MONOTOPIC MEMBRANE PROTEINS",
    "monotopic proteins": "MONOTOPIC MEMBRANE PROTEINS",
    "monotopic/peripheral": "MONOTOPIC MEMBRANE PROTEINS",
    "all alpha monotopic/peripheral": "MONOTOPIC MEMBRANE PROTEINS",
    "peripheral": "MONOTOPIC MEMBRANE PROTEINS",
    "bitopic": "BITOPIC PROTEINS",
    "bitopic proteins": "BITOPIC PROTEINS",
    "transmembrane proteins:alpha-helical": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
    "transmembrane protein:alpha-helical": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
    "alpha-helical": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
    "alpha helical": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
    "all alpha": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
    "transmembrane proteins:beta-barrel": "TRANSMEMBRANE PROTEINS:BETA-BARREL",
    "transmembrane protein:beta-barrel": "TRANSMEMBRANE PROTEINS:BETA-BARREL",
    "beta-barrel": "TRANSMEMBRANE PROTEINS:BETA-BARREL",
    "beta barrel": "TRANSMEMBRANE PROTEINS:BETA-BARREL",
}

DISCREPANCY_EQUIVALENT_GROUPS = {
    "MONOTOPIC MEMBRANE PROTEINS": "MONOTOPIC MEMBRANE PROTEINS",
    "BITOPIC PROTEINS": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL / BITOPIC",
    "TRANSMEMBRANE": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL / BITOPIC",
    "TRANSMEMBRANE PROTEINS": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL / BITOPIC",
    "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL / BITOPIC",
}

EXPERT_BENCHMARK_GROUPS = {
    "MONOTOPIC MEMBRANE PROTEINS": "MONOTOPIC MEMBRANE PROTEINS",
    "BITOPIC PROTEINS": "TRANSMEMBRANE / BITOPIC BENCHMARK",
    "TRANSMEMBRANE": "TRANSMEMBRANE / BITOPIC BENCHMARK",
    "TRANSMEMBRANE PROTEINS": "TRANSMEMBRANE / BITOPIC BENCHMARK",
    "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL": "TRANSMEMBRANE / BITOPIC BENCHMARK",
    "TRANSMEMBRANE PROTEINS:BETA-BARREL": "TRANSMEMBRANE / BITOPIC BENCHMARK",
}


def standardize_group_label(value):
    normalized = str(value or "").strip()
    if not normalized:
        return None

    lowered = re.sub(r"\s+", " ", normalized).strip().lower()
    canonical = CANONICAL_GROUP_LABELS.get(lowered)
    if canonical:
        return canonical

    if "monotopic" in lowered or "peripheral" in lowered:
        return "MONOTOPIC MEMBRANE PROTEINS"
    if "bitopic" in lowered:
        return "BITOPIC PROTEINS"
    if "beta" in lowered and "barrel" in lowered:
        return "TRANSMEMBRANE PROTEINS:BETA-BARREL"
    if "alpha" in lowered and ("helical" in lowered or "helix" in lowered):
        return "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL"

    return normalized.upper()


def collapse_group_label_for_disagreement(value):
    standardized = standardize_group_label(value)
    if not standardized:
        return None

    direct = DISCREPANCY_EQUIVALENT_GROUPS.get(standardized)
    if direct:
        return direct

    lowered = standardized.lower()
    if "monotopic" in lowered or "peripheral" in lowered:
        return "MONOTOPIC MEMBRANE PROTEINS"
    if "bitopic" in lowered:
        return "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL / BITOPIC"
    if "transmembrane" in lowered and "beta" not in lowered:
        return "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL / BITOPIC"
    if "alpha" in lowered and ("helical" in lowered or "helix" in lowered):
        return "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL / BITOPIC"

    return standardized


def collapse_group_label_for_expert_benchmark(value):
    standardized = standardize_group_label(value)
    if not standardized:
        return None

    direct = EXPERT_BENCHMARK_GROUPS.get(standardized)
    if direct:
        return direct

    lowered = standardized.lower()
    if "monotopic" in lowered or "peripheral" in lowered:
        return "MONOTOPIC MEMBRANE PROTEINS"
    if "bitopic" in lowered:
        return "TRANSMEMBRANE / BITOPIC BENCHMARK"
    if "transmembrane" in lowered:
        return "TRANSMEMBRANE / BITOPIC BENCHMARK"
    if "alpha" in lowered and ("helical" in lowered or "helix" in lowered):
        return "TRANSMEMBRANE / BITOPIC BENCHMARK"
    if "beta" in lowered and "barrel" in lowered:
        return "TRANSMEMBRANE / BITOPIC BENCHMARK"

    return standardized


def parse_tm_count(value):
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None

    try:
        return int(float(normalized))
    except (TypeError, ValueError):
        match = re.search(r"-?\d+", normalized)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                return None
    return None
