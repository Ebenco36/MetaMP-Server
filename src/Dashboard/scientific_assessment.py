from __future__ import annotations

import re
from typing import Any


# Explicit, curated overrides for known edge cases.
# These are treated as the strongest evidence source.
ENTRY_FLAG_OVERRIDES = {
    "1PFO": {
        "state_dependent": True,
        "sequence_only_topology_benchmark_suitable": False,
        "soluble_to_membrane_transition": True,
        "confidence": "high",
        "notes": [
            "Perfringolysin O is a pore-forming toxin with soluble and membrane-inserted states.",
        ],
        "matched_rule_ids": ["override:1PFO"],
    },
    "1YGM": {
        "not_membrane_protein_candidate": True,
        "sequence_only_topology_benchmark_suitable": False,
        "confidence": "high",
        "notes": [
            "Mistic is commonly treated as a membrane-associated expression helper rather than a canonical membrane protein.",
        ],
        "matched_rule_ids": ["override:1YGM"],
    },
}


# Keyword rules are intentionally split into harder exclusion patterns and softer
# suspicion patterns. Weak rules can contribute confidence and review signals
# without automatically making an entry benchmark-ineligible on their own.
KEYWORD_FLAG_RULES = (
    {
        "id": "pore_forming_toxin",
        "pattern": re.compile(
            r"\b(pore[- ]forming|hemolysin|cytolysin|perfringolysin|cytolytic|pore former|"
            r"cholesterol[- ]dependent cytolysin|membrane attack complex|macpf|aerolysin|anthrolysin|"
            r"streptolysin|listeriolysin|leukocidin|tox[in]?)\b",
            re.IGNORECASE,
        ),
        "flags": {
            "state_dependent": True,
            "soluble_to_membrane_transition": True,
        },
        "benchmark_override": False,
        "confidence": "high",
        "reason": "Pore-forming or toxin-like terminology suggests a soluble-to-membrane transition or state-dependent topology.",
    },
    {
        "id": "assembly_dependent_insertion",
        "pattern": re.compile(
            r"\b(prepore|pre[- ]pore|oligomer(?:ic|ization)?|multimer(?:ic|ization)?|assembly[- ]dependent|"
            r"membrane insertion|inserted state|insertion complex|pore assembly)\b",
            re.IGNORECASE,
        ),
        "flags": {
            "state_dependent": True,
        },
        "benchmark_override": False,
        "confidence": "medium",
        "reason": "Assembly- or oligomerization-dependent wording suggests topology may depend on structural state.",
    },
    {
        "id": "explicit_state_switching",
        "pattern": re.compile(
            r"\b(state[- ]dependent|conformational change|conformational switch|switching|activated state|"
            r"inactive state|transition state|closed state|open state)\b",
            re.IGNORECASE,
        ),
        "flags": {
            "state_dependent": True,
        },
        "benchmark_override": None,
        "confidence": "medium",
        "reason": "Record text suggests multiple structural states that can affect topology interpretation.",
    },
    {
        "id": "amphitropic_or_reversible_association",
        "pattern": re.compile(
            r"\b(amphitropic|peripheral membrane(?: protein)?|membrane[- ]associated|reversible membrane binding|"
            r"lipid[- ]binding|membrane recruitment)\b",
            re.IGNORECASE,
        ),
        "flags": {
            "state_dependent": True,
        },
        "benchmark_override": None,
        "confidence": "medium",
        "reason": "Text suggests reversible or context-dependent membrane association rather than fixed embedded topology.",
    },
    {
        "id": "expression_helper_or_fusion_construct",
        "pattern": re.compile(
            r"\b(fusion partner|expression helper|mistic|carrier protein|solubility tag|fusion construct)\b",
            re.IGNORECASE,
        ),
        "flags": {
            "not_membrane_protein_candidate": True,
        },
        "benchmark_override": False,
        "confidence": "high",
        "reason": "Likely engineered helper, fusion construct, or non-canonical membrane-associated tool protein.",
    },
    {
        "id": "engineered_or_partial_construct",
        "pattern": re.compile(
            r"\b(truncated|engineered|chimera|chimeric|synthetic construct|designed protein|isolated domain|"
            r"partial structure|protein fragment|fragment)\b",
            re.IGNORECASE,
        ),
        "flags": {},
        "benchmark_override": None,
        "confidence": "low",
        "reason": "Construct appears engineered or partial, so deposited topology may not represent the full native system.",
    },
    {
        "id": "artificial_environment_context",
        "pattern": re.compile(
            r"\b(detergent|micelle|nanodisc|bicelle|liposome)\b",
            re.IGNORECASE,
        ),
        "flags": {},
        "benchmark_override": None,
        "confidence": "low",
        "reason": "Structure was described in an artificial membrane-mimetic context; interpret topology cautiously.",
    },
)

_CONFIDENCE_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _normalize_replacement_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"replaced", "true", "1", "yes", "y"}:
        return True
    if normalized in {"not replaced", "false", "0", "no", "n", ""}:
        return False
    return False


def _normalize_bool_like(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _append_confidence(confidence_scores: list[str], value: Any) -> None:
    normalized = str(value or "").strip().lower()
    if normalized in _CONFIDENCE_ORDER and normalized != "none":
        confidence_scores.append(normalized)


def _set_benchmark_override(current: bool | None, candidate: bool | None) -> bool | None:
    """Combine benchmark recommendations conservatively.

    False is sticky because exclusion evidence should not be undone by a weaker,
    later rule. True is used only when no prior decision exists.
    """
    if candidate is None:
        return current
    if current is False:
        return False
    if candidate is False:
        return False
    if current is None:
        return candidate
    return current


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


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

    raw_flags: dict[str, Any] = {
        "state_dependent": False,
        "soluble_to_membrane_transition": False,
        "not_membrane_protein_candidate": False,
        "complex_ambiguity": bool((structure_context.get("chain_count") or 0) > 1),
        "obsolete_or_replaced": _normalize_replacement_flag(record.get("is_replaced")),
        "confidence_scores": [],
        "matched_rule_ids": [],
        "soft_review_reasons": [],
    }
    benchmark_recommended_override: bool | None = None
    explicit_benchmark_override = False
    notes: list[str] = []

    override = ENTRY_FLAG_OVERRIDES.get(pdb_code)
    if override:
        for key, value in override.items():
            if key == "notes":
                notes.extend(value)
            elif key == "matched_rule_ids":
                raw_flags["matched_rule_ids"].extend(list(value))
            elif key == "confidence":
                _append_confidence(raw_flags["confidence_scores"], value)
            elif key == "sequence_only_topology_benchmark_suitable":
                benchmark_recommended_override = _normalize_bool_like(value)
                explicit_benchmark_override = benchmark_recommended_override is not None
            else:
                raw_flags[key] = value

    for rule in KEYWORD_FLAG_RULES:
        if not text_blob or not rule["pattern"].search(text_blob):
            continue

        raw_flags["matched_rule_ids"].append(rule["id"])
        reason = rule.get("reason")
        if reason:
            notes.append(str(reason))

        _append_confidence(raw_flags["confidence_scores"], rule.get("confidence"))
        if not explicit_benchmark_override:
            benchmark_recommended_override = _set_benchmark_override(
                benchmark_recommended_override,
                _normalize_bool_like(rule.get("benchmark_override")),
            )

        for key, value in (rule.get("flags") or {}).items():
            if isinstance(value, bool):
                raw_flags[key] = bool(raw_flags.get(key, False) or value)
            else:
                raw_flags[key] = value

        if rule.get("benchmark_override") is None and reason:
            raw_flags["soft_review_reasons"].append(str(reason))

    if raw_flags["obsolete_or_replaced"]:
        notes.append(
            "Entry has replacement/obsolescence metadata and should be interpreted through its canonical record."
        )
    if raw_flags["complex_ambiguity"]:
        notes.append(
            "Entry contains multi-chain or multi-entity context that can complicate topology interpretation."
        )

    context_reasons: list[str] = []
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

    confidence_scores = raw_flags.get("confidence_scores", [])
    if confidence_scores:
        overall_confidence = max(confidence_scores, key=lambda item: _CONFIDENCE_ORDER.get(item, 0))
    else:
        overall_confidence = "none"

    review_recommended = bool(raw_flags["soft_review_reasons"]) or flags["multichain_context"]

    recommended_for_sequence_topology_benchmark = (
        benchmark_recommended_override
        if benchmark_recommended_override is not None
        else not flags["context_dependent_topology"] and not flags["non_canonical_membrane_case"]
    )
    recommended_for_sequence_topology_benchmark = (
        bool(recommended_for_sequence_topology_benchmark)
        and not flags["obsolete_or_replaced"]
    )

    benchmark_exclusion_reasons: list[str] = []
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
            "matched_rule_ids": _dedupe_preserve_order(raw_flags.get("matched_rule_ids", [])),
            "soft_review_reasons": _dedupe_preserve_order(raw_flags.get("soft_review_reasons", [])),
        },
        "notes": _dedupe_preserve_order(notes),
        "benchmark_exclusion_reasons": benchmark_exclusion_reasons,
        "recommended_for_sequence_topology_benchmark": recommended_for_sequence_topology_benchmark,
        "confidence": overall_confidence,
        "review_recommended": review_recommended,
    }
