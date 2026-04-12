import logging
import math

import pandas as pd
from sqlalchemy import or_

from database.db import db
from src.MP.model import MembraneProteinData
from src.MP.model_pdb import PDB


logger = logging.getLogger(__name__)


def normalize_pdb_code(value):
    text = str(value or "").strip().upper()
    return text or ""


def normalize_replacement_flag(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return False
        return bool(int(value))
    text = str(value).strip().lower()
    if not text:
        return False
    return text in {"1", "true", "yes", "y", "replaced", "obsolete"}


def resolve_replacement_aliases(pdb_codes=None):
    selected_codes = {
        normalize_pdb_code(code)
        for code in (pdb_codes or ())
        if normalize_pdb_code(code)
    }
    if not selected_codes:
        return {}

    alias_map = {code: {code} for code in selected_codes}

    rows = []
    for model in (PDB, MembraneProteinData):
        try:
            model_rows = (
                db.session.query(
                    model.pdb_code,
                    model.replacement_pdb_code,
                    model.canonical_pdb_code,
                    model.is_replaced,
                )
                .filter(
                    or_(
                        db.func.upper(db.func.trim(model.pdb_code)).in_(selected_codes),
                        db.func.upper(db.func.trim(model.replacement_pdb_code)).in_(selected_codes),
                        db.func.upper(db.func.trim(model.canonical_pdb_code)).in_(selected_codes),
                    )
                )
                .all()
            )
        except Exception as exc:
            logger.warning(
                "Unable to load replacement aliases from %s for %s: %s",
                getattr(model, "__tablename__", getattr(model, "__name__", "unknown")),
                sorted(selected_codes),
                exc,
            )
            continue
        rows.extend(model_rows)

    for row in rows:
        pdb_code = normalize_pdb_code(getattr(row, "pdb_code", None))
        replacement_pdb_code = normalize_pdb_code(getattr(row, "replacement_pdb_code", None))
        canonical_pdb_code = normalize_pdb_code(getattr(row, "canonical_pdb_code", None))
        is_replaced = normalize_replacement_flag(getattr(row, "is_replaced", None))
        effective_code = canonical_pdb_code or replacement_pdb_code or pdb_code
        aliases = {value for value in (pdb_code, replacement_pdb_code, canonical_pdb_code) if value}
        if not aliases:
            continue
        if not effective_code:
            effective_code = pdb_code
        aliases.add(effective_code)
        for alias in aliases:
            alias_map.setdefault(alias, set()).update(aliases)
            if is_replaced and effective_code:
                alias_map[alias].add(effective_code)

    return {key: sorted(values) for key, values in alias_map.items()}


def resolve_canonical_pdb_code(pdb_code):
    normalized_code = normalize_pdb_code(pdb_code)
    if not normalized_code:
        return ""

    for model in (PDB, MembraneProteinData):
        try:
            row = (
                db.session.query(
                    model.pdb_code,
                    model.replacement_pdb_code,
                    model.canonical_pdb_code,
                    model.is_replaced,
                )
                .filter(
                    or_(
                        db.func.upper(db.func.trim(model.pdb_code)) == normalized_code,
                        db.func.upper(db.func.trim(model.replacement_pdb_code)) == normalized_code,
                        db.func.upper(db.func.trim(model.canonical_pdb_code)) == normalized_code,
                    )
                )
                .first()
            )
        except Exception as exc:
            logger.warning(
                "Unable to resolve canonical PDB code for %s via %s: %s",
                normalized_code,
                getattr(model, "__tablename__", getattr(model, "__name__", "unknown")),
                exc,
            )
            continue
        if row is None:
            continue

        pdb_code_value = normalize_pdb_code(getattr(row, "pdb_code", None))
        replacement_pdb_code = normalize_pdb_code(getattr(row, "replacement_pdb_code", None))
        canonical_pdb_code = normalize_pdb_code(getattr(row, "canonical_pdb_code", None))
        is_replaced = normalize_replacement_flag(getattr(row, "is_replaced", None))

        if canonical_pdb_code:
            return canonical_pdb_code
        if replacement_pdb_code:
            return replacement_pdb_code
        if is_replaced and pdb_code_value and pdb_code_value != normalized_code:
            return pdb_code_value
        if pdb_code_value:
            return pdb_code_value

    return normalized_code


def canonicalize_pdb_codes(pdb_codes=None):
    normalized_codes = [
        normalize_pdb_code(code)
        for code in (pdb_codes or ())
        if normalize_pdb_code(code)
    ]
    if not normalized_codes:
        return []

    canonicalized = []
    for code in normalized_codes:
        canonicalized.append(resolve_canonical_pdb_code(code) or code)
    return list(dict.fromkeys(canonicalized))


def canonicalize_pdb_frame(
    frame,
    pdb_column="pdb_code",
    canonical_columns=("canonical_pdb_code", "replacement_pdb_code"),
    is_replaced_column="is_replaced",
):
    if frame is None or frame.empty or pdb_column not in frame.columns:
        return frame

    result = frame.copy()
    original_codes = result[pdb_column].fillna("").astype(str).str.strip().str.upper()
    canonical_series = pd.Series("", index=result.index, dtype=str)

    for column in canonical_columns:
        if column in result.columns:
            candidate = result[column].fillna("").astype(str).str.strip().str.upper()
            canonical_series = canonical_series.mask(canonical_series.eq(""), candidate)

    lookup_codes = list(dict.fromkeys(
        [
            value
            for value in pd.concat([original_codes, canonical_series]).tolist()
            if str(value or "").strip()
        ]
    ))
    canonical_map = {
        code: resolve_canonical_pdb_code(code) or code
        for code in lookup_codes
    }

    resolved_codes = []
    for index, original_code in original_codes.items():
        preferred_code = canonical_series.loc[index] or original_code
        resolved_codes.append(canonical_map.get(preferred_code) or canonical_map.get(original_code) or original_code)

    result["_original_pdb_code"] = original_codes
    result[pdb_column] = resolved_codes

    replacement_rank = pd.Series(1, index=result.index, dtype=int)
    if is_replaced_column in result.columns:
        replacement_rank = result[is_replaced_column].apply(
            lambda value: 2 if normalize_replacement_flag(value) else 0
        )
    else:
        replacement_rank = (
            result["_original_pdb_code"].fillna("").astype(str).str.strip().str.upper()
            != result[pdb_column].fillna("").astype(str).str.strip().str.upper()
        ).astype(int)

    sequence_rank = pd.Series(0, index=result.index, dtype=int)
    if "sequence_sequence" in result.columns:
        sequence_rank = (
            result["sequence_sequence"].fillna("").astype(str).str.strip().ne("")
        ).astype(int) * -1

    result["_replacement_rank"] = replacement_rank
    result["_sequence_rank"] = sequence_rank
    result = result.sort_values(
        by=[pdb_column, "_replacement_rank", "_sequence_rank", "_original_pdb_code"],
        ascending=[True, True, True, True],
        kind="stable",
    )
    result = result.drop_duplicates(subset=pdb_column, keep="first").copy()
    return result.drop(columns=["_original_pdb_code", "_replacement_rank", "_sequence_rank"], errors="ignore")
