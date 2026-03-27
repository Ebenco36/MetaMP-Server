from __future__ import annotations

import re


ENTRY_FLAG_OVERRIDES = {
    "1PFO": {
        "state_dependent": True,
        "sequence_only_topology_benchmark_suitable": False,
        "soluble_to_membrane_transition": True,
        "notes": [
            "Perfringolysin O is a pore-forming toxin with soluble and membrane-inserted states.",
        ],
    },
    "1YGM": {
        "not_membrane_protein_candidate": True,
        "sequence_only_topology_benchmark_suitable": False,
        "notes": [
            "Mistic is commonly treated as a membrane-associated expression helper rather than a canonical membrane protein.",
        ],
    },
}

KEYWORD_FLAG_RULES = (
    (
        re.compile(r"\b(pore[- ]forming|hemolysin|cytolysin|perfringolysin|toxin)\b", re.IGNORECASE),
        {
            "state_dependent": True,
            "sequence_only_topology_benchmark_suitable": False,
            "soluble_to_membrane_transition": True,
            "reason": "Name/title suggests a pore-forming or toxin-like soluble-to-membrane transition.",
        },
    ),
    (
        re.compile(r"\b(fusion partner|expression helper|mistic)\b", re.IGNORECASE),
        {
            "not_membrane_protein_candidate": True,
            "sequence_only_topology_benchmark_suitable": False,
            "reason": "Name/title suggests an expression helper or fusion-partner construct.",
        },
    ),
)


def _normalize_replacement_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"replaced", "true", "1", "yes", "y"}:
        return True
    if normalized in {"not replaced", "false", "0", "no", "n", ""}:
        return False
    return False


def build_scientific_assessment(record: dict | None) -> dict:
    record = record or {}
    pdb_code = str(
        record.get("canonical_pdb_code")
        or record.get("pdb_code")
        or record.get("PDB Code")
        or ""
    ).strip().upper()
    text_blob = " ".join(
        str(record.get(key) or "")
        for key in ("name", "struct_title", "protein_recommended_name", "description")
    ).strip()
    structure_context = record.get("structure_context") or {}

    raw_flags = {
        "state_dependent": False,
        "soluble_to_membrane_transition": False,
        "not_membrane_protein_candidate": False,
        "complex_ambiguity": bool((structure_context.get("chain_count") or 0) > 1),
        "obsolete_or_replaced": _normalize_replacement_flag(record.get("is_replaced")),
    }
    benchmark_recommended_override = None
    notes = []

    override = ENTRY_FLAG_OVERRIDES.get(pdb_code)
    if override:
        for key, value in override.items():
            if key == "notes":
                notes.extend(value)
            elif key == "sequence_only_topology_benchmark_suitable":
                benchmark_recommended_override = bool(value)
            else:
                raw_flags[key] = value

    for pattern, outcome in KEYWORD_FLAG_RULES:
        if text_blob and pattern.search(text_blob):
            for key, value in outcome.items():
                if key == "reason":
                    notes.append(value)
                elif key == "sequence_only_topology_benchmark_suitable":
                    benchmark_recommended_override = bool(value)
                else:
                    raw_flags[key] = value

    if raw_flags["obsolete_or_replaced"]:
        notes.append("Entry has replacement/obsolescence metadata and should be interpreted through its canonical record.")
    if raw_flags["complex_ambiguity"]:
        notes.append("Entry contains multi-chain or multi-entity context that can complicate topology interpretation.")

    context_reasons = []
    if raw_flags["state_dependent"]:
        context_reasons.append("state_dependent")
    if raw_flags["soluble_to_membrane_transition"]:
        context_reasons.append("soluble_to_membrane_transition")

    flags = {
        "context_dependent_topology": bool(context_reasons),
        "non_canonical_membrane_case": bool(raw_flags["not_membrane_protein_candidate"]),
        "multichain_context": bool(raw_flags["complex_ambiguity"]),
        "obsolete_or_replaced": bool(raw_flags["obsolete_or_replaced"]),
    }

    recommended_for_sequence_topology_benchmark = (
        benchmark_recommended_override
        if benchmark_recommended_override is not None
        else not flags["context_dependent_topology"] and not flags["non_canonical_membrane_case"]
    )
    recommended_for_sequence_topology_benchmark = (
        bool(recommended_for_sequence_topology_benchmark)
        and not flags["obsolete_or_replaced"]
    )

    benchmark_exclusion_reasons = []
    if flags["context_dependent_topology"]:
        benchmark_exclusion_reasons.append("context_dependent_topology")
    if flags["non_canonical_membrane_case"]:
        benchmark_exclusion_reasons.append("non_canonical_membrane_case")
    if flags["obsolete_or_replaced"]:
        benchmark_exclusion_reasons.append("replaced_entry")
    if (
        not recommended_for_sequence_topology_benchmark
        and not benchmark_exclusion_reasons
    ):
        benchmark_exclusion_reasons.append("sequence_topology_not_recommended")

    return {
        "flags": flags,
        "details": {
            "context_reasons": context_reasons,
        },
        "notes": list(dict.fromkeys(notes)),
        "benchmark_exclusion_reasons": benchmark_exclusion_reasons,
        "recommended_for_sequence_topology_benchmark": recommended_for_sequence_topology_benchmark,
    }
