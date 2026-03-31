import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

from flask import current_app, has_app_context
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import text

from database.db import db
from src.AI_Packages.TMAlphaFoldPredictorClient import (
    TMALPHAFOLD_AUX_METHODS,
    TMALPHAFOLD_PROVIDER,
    TMALPHAFOLD_SEQUENCE_METHODS,
    TMAlphaFoldPredictionResult,
    _extract_sequence_from_payload,
    TMAlphaFoldPredictorClient,
)
from src.AI_Packages.TMProteinPredictor import normalize_tm_regions_json_string
from src.MP.model_tmalphafold import TMAlphaFoldPrediction
from src.MP.model_uniprot import Uniprot


def _emit_progress(progress_callback, message):
    if progress_callback is not None:
        progress_callback(message)


def _ensure_tmalphafold_storage(progress_callback=None):
    TMAlphaFoldPrediction.__table__.create(bind=db.engine, checkfirst=True)
    current_app.logger.info("TMAlphaFold table ensured (created if not existed).") 
    _emit_progress(progress_callback, "Ensured TMAlphaFold prediction storage is available.")


def _normalize_selection(pdb_codes=None):
    normalized = []
    for value in pdb_codes or ():
        text = str(value or "").strip().upper()
        if text:
            normalized.append(text)
    return list(dict.fromkeys(normalized))


def _normalize_tm_count_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null", "na", "n/a"}:
        return None
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return int(numeric)


def _normalize_tm_regions_json_value(value):
    if value is None:
        return "[]"
    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(list(value))
        except (TypeError, ValueError):
            return "[]"
    return normalize_tm_regions_json_string(value)


def normalize_tmalphafold_methods(methods=None):
    if not methods:
        return list(TMALPHAFOLD_SEQUENCE_METHODS + TMALPHAFOLD_AUX_METHODS)
    normalized = []
    for method in methods:
        text = str(method or "").strip()
        if not text:
            continue
        for candidate in TMALPHAFOLD_SEQUENCE_METHODS + TMALPHAFOLD_AUX_METHODS:
            if candidate.lower() == text.lower():
                normalized.append(candidate)
                break
        else:
            raise ValueError(
                f"Unsupported TMAlphaFold method '{method}'. "
                f"Expected one of: {', '.join(TMALPHAFOLD_SEQUENCE_METHODS + TMALPHAFOLD_AUX_METHODS)}."
            )
    return list(dict.fromkeys(normalized))


def load_tmalphafold_targets(pdb_codes=None, limit=None):
    query = db.session.query(Uniprot.pdb_code, Uniprot.uniprot_id).filter(
        Uniprot.pdb_code.isnot(None),
        Uniprot.uniprot_id.isnot(None),
    )
    selected = _normalize_selection(pdb_codes)
    if selected:
        query = query.filter(db.func.upper(db.func.trim(Uniprot.pdb_code)).in_(selected))

    rows = (
        query.distinct(Uniprot.pdb_code, Uniprot.uniprot_id)
        .order_by(Uniprot.pdb_code.asc(), Uniprot.uniprot_id.asc())
        .all()
    )
    targets = [
        {"pdb_code": str(pdb_code).strip().upper(), "uniprot_id": str(uniprot_id).strip().upper()}
        for pdb_code, uniprot_id in rows
        if str(pdb_code or "").strip() and str(uniprot_id or "").strip()
    ]
    if limit is not None:
        targets = targets[: int(limit)]
    return targets


def _upsert_predictions(predictions):
    if not predictions:
        return {"stored_rows": 0, "inserted_rows": 0, "updated_rows": 0}

    payload = [
        {
            "pdb_code": item.pdb_code,
            "uniprot_id": item.uniprot_id,
            "provider": item.provider,
            "method": item.method,
            "prediction_kind": item.prediction_kind,
            "tm_count": _normalize_tm_count_value(item.tm_count),
            "tm_regions_json": _normalize_tm_regions_json_value(item.tm_regions_json),
            "raw_payload_json": item.raw_payload_json,
            "source_url": item.source_url,
            "status": item.status,
            "error_message": item.error_message,
        }
        for item in predictions
    ]
    keys = [
        (
            str(item["provider"] or "").strip(),
            str(item["method"] or "").strip(),
            str(item["pdb_code"] or "").strip().upper(),
            str(item["uniprot_id"] or "").strip().upper(),
        )
        for item in payload
    ]
    unique_keys = list(dict.fromkeys(keys))

    existing_query = TMAlphaFoldPrediction.query.with_entities(
        TMAlphaFoldPrediction.provider,
        TMAlphaFoldPrediction.method,
        TMAlphaFoldPrediction.pdb_code,
        TMAlphaFoldPrediction.uniprot_id,
    ).filter(
        db.tuple_(
            TMAlphaFoldPrediction.provider,
            TMAlphaFoldPrediction.method,
            db.func.upper(db.func.trim(TMAlphaFoldPrediction.pdb_code)),
            db.func.upper(db.func.trim(TMAlphaFoldPrediction.uniprot_id)),
        ).in_(unique_keys)
    )
    existing_key_count = sum(1 for _ in existing_query)

    bind = db.engine
    if bind.dialect.name == "postgresql":
        try:
            stmt = pg_insert(TMAlphaFoldPrediction.__table__).values(payload)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_tmalphafold_provider_method_pdb_uniprot",
                set_={
                    "prediction_kind": stmt.excluded.prediction_kind,
                    "tm_count": stmt.excluded.tm_count,
                    "tm_regions_json": stmt.excluded.tm_regions_json,
                    "raw_payload_json": stmt.excluded.raw_payload_json,
                    "source_url": stmt.excluded.source_url,
                    "status": stmt.excluded.status,
                    "error_message": stmt.excluded.error_message,
                    "updated_at": db.func.now(),
                },
            )
            db.session.execute(stmt)
            db.session.commit()

            inserted_rows = max(0, len(payload) - existing_key_count)
            updated_rows = min(existing_key_count, len(payload))
            current_app.logger.info(
                f"[PG UPSERT] Committed {len(payload)} row(s): "
                f"~{inserted_rows} inserted, ~{updated_rows} updated."
            )

        except Exception as exc:
            db.session.rollback()
            current_app.logger.error(
                f"[PG UPSERT] Failed and rolled back: {exc}",
                exc_info=True,
            )
            raise
        return {
            "stored_rows": len(payload),
            "inserted_rows": inserted_rows,
            "updated_rows": updated_rows,
        }

    try:
        for item in payload:
            row = TMAlphaFoldPrediction.query.filter_by(
                provider=item["provider"],
                method=item["method"],
                pdb_code=item["pdb_code"],
                uniprot_id=item["uniprot_id"],
            ).first()
            if row is None:
                row = TMAlphaFoldPrediction(**item)
                db.session.add(row)
                current_app.logger.debug(f"[INSERT] New row: {item['pdb_code']} / {item['uniprot_id']} / {item['method']}")
            else:
                for key, value in item.items():
                    setattr(row, key, value)
                current_app.logger.debug(f"[UPDATE] Existing row: {item['pdb_code']} / {item['uniprot_id']} / {item['method']}")
        db.session.commit()
        current_app.logger.info(f"[FALLBACK UPSERT] Committed {len(payload)} row(s).")
    except Exception as exc:
        db.session.rollback()
        current_app.logger.error(
            f"[FALLBACK UPSERT] Failed and rolled back: {exc}",
            exc_info=True,
        )
        raise
    inserted_rows = max(0, len(payload) - existing_key_count)
    updated_rows = min(existing_key_count, len(payload))
    return {
        "stored_rows": len(payload),
        "inserted_rows": inserted_rows,
        "updated_rows": updated_rows,
    }


def _lookup_uniprot_ids_for_pdb_codes(pdb_codes):
    normalized_codes = {
        str(code or "").strip().upper()
        for code in pdb_codes or ()
        if str(code or "").strip()
    }
    if not normalized_codes:
        return {}

    rows = (
        db.session.query(Uniprot.pdb_code, Uniprot.uniprot_id)
        .filter(
            Uniprot.pdb_code.isnot(None),
            Uniprot.uniprot_id.isnot(None),
            db.func.upper(db.func.trim(Uniprot.pdb_code)).in_(normalized_codes),
        )
        .all()
    )
    mapping = defaultdict(set)
    for pdb_code, uniprot_id in rows:
        normalized_pdb = str(pdb_code or "").strip().upper()
        normalized_uniprot = str(uniprot_id or "").strip().upper()
        if normalized_pdb and normalized_uniprot:
            mapping[normalized_pdb].add(normalized_uniprot)
    return {key: sorted(values) for key, values in mapping.items()}


def mirror_local_tm_prediction_rows(
    method,
    records,
    provider="MetaMP",
    prediction_kind="sequence_topology",
    progress_callback=None,
):
    normalized_records_by_pdb = {}
    for row in records or ():
        pdb_code = str((row or {}).get("pdb_code") or "").strip().upper()
        if not pdb_code:
            continue
        tm_count = _normalize_tm_count_value((row or {}).get(f"{method}_tm_count"))
        tm_regions_json = _normalize_tm_regions_json_value((row or {}).get(f"{method}_tm_regions"))
        if tm_count is None and tm_regions_json == "[]":
            continue
        normalized_records_by_pdb[pdb_code] = {
            "pdb_code": pdb_code,
            "tm_count": tm_count,
            "tm_regions_json": tm_regions_json,
        }
    normalized_records = list(normalized_records_by_pdb.values())
    if not normalized_records:
        return {"stored_rows": 0, "provider": provider, "method": method}

    _ensure_tmalphafold_storage(progress_callback=progress_callback)
    uniprot_mapping = _lookup_uniprot_ids_for_pdb_codes(
        [item["pdb_code"] for item in normalized_records]
    )

    payloads = []
    for item in normalized_records:
        pdb_code = item["pdb_code"]
        uniprot_ids = list(dict.fromkeys(uniprot_mapping.get(pdb_code) or [f"PDB:{pdb_code}"]))
        raw_payload = json.dumps(
            {
                "provider": provider,
                "method": method,
                "pdb_code": pdb_code,
                "tm_count": item["tm_count"],
                "tm_regions_json": item["tm_regions_json"],
                "uniprot_ids": uniprot_ids,
            }
        )
        for uniprot_id in uniprot_ids:
            payloads.append(
                TMAlphaFoldPredictionResult(
                    pdb_code=pdb_code,
                    uniprot_id=uniprot_id,
                    provider=provider,
                    method=method,
                    prediction_kind=prediction_kind,
                    tm_count=item["tm_count"],
                    tm_regions_json=item["tm_regions_json"],
                    raw_payload_json=raw_payload,
                    source_url="",
                    status="success",
                    sequence_sequence=None,
                    error_message=None,
                )
            )

    stored_rows = _upsert_predictions(payloads)
    _emit_progress(
        progress_callback,
        (
            f"Mirrored {stored_rows['stored_rows']} normalized {method} prediction row(s) into "
            "membrane_protein_tmalphafold_predictions "
            f"({stored_rows['inserted_rows']} inserted, {stored_rows['updated_rows']} updated)."
        ),
    )
    return {
        "stored_rows": stored_rows["stored_rows"],
        "inserted_rows": stored_rows["inserted_rows"],
        "updated_rows": stored_rows["updated_rows"],
        "provider": provider,
        "method": method,
    }


def _collect_sequence_backfills(predictions):
    sequence_by_target = {}
    conflicting_targets = set()
    for item in predictions or ():
        if item.status != "success":
            continue
        sequence = str(item.sequence_sequence or "").strip()
        if not sequence:
            continue
        key = (str(item.pdb_code or "").strip().upper(), str(item.uniprot_id or "").strip().upper())
        if not all(key):
            continue
        existing = sequence_by_target.get(key)
        if existing is None:
            sequence_by_target[key] = sequence
            continue
        if existing != sequence:
            conflicting_targets.add(key)

    rows = [
        {"pdb_code": pdb_code, "uniprot_id": uniprot_id, "sequence_sequence": sequence}
        for (pdb_code, uniprot_id), sequence in sequence_by_target.items()
        if (pdb_code, uniprot_id) not in conflicting_targets
    ]
    return {
        "rows": rows,
        "conflicting_targets": sorted(conflicting_targets),
    }


def _collect_sequence_backfills_from_stored_rows(rows):
    candidates = []
    for row in rows or ():
        raw_payload = str(row.raw_payload_json or "").strip()
        if not raw_payload:
            continue
        try:
            payload = json.loads(raw_payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        sequence = _extract_sequence_from_payload(payload)
        if not sequence:
            continue
        candidates.append(
            SimpleNamespace(
                status="success",
                pdb_code=row.pdb_code,
                uniprot_id=row.uniprot_id,
                sequence_sequence=sequence,
            )
        )
    return _collect_sequence_backfills(candidates)


def _persist_missing_sequences(sequence_rows, progress_callback=None):
    rows = [
        {
            "pdb_code": str(item.get("pdb_code") or "").strip().upper(),
            "uniprot_id": str(item.get("uniprot_id") or "").strip().upper(),
            "sequence_sequence": str(item.get("sequence_sequence") or "").strip(),
        }
        for item in sequence_rows or ()
        if str(item.get("pdb_code") or "").strip()
        and str(item.get("uniprot_id") or "").strip()
        and str(item.get("sequence_sequence") or "").strip()
    ]
    if not rows:
        return {
            "protein_rows_updated": 0,
            "uniprot_rows_updated": 0,
            "attempted_targets": 0,
        }

    protein_stmt = text(
        """
        UPDATE membrane_proteins
        SET sequence_sequence = :sequence_sequence
        WHERE UPPER(TRIM(pdb_code)) = :pdb_code
          AND (sequence_sequence IS NULL OR BTRIM(sequence_sequence) = '')
        """
    )
    uniprot_stmt = text(
        """
        UPDATE membrane_protein_uniprot
        SET sequence_sequence = :sequence_sequence
        WHERE UPPER(TRIM(pdb_code)) = :pdb_code
          AND UPPER(TRIM(uniprot_id)) = :uniprot_id
          AND (sequence_sequence IS NULL OR BTRIM(sequence_sequence) = '')
        """
    )

    protein_result = db.session.execute(protein_stmt, rows)
    uniprot_result = db.session.execute(uniprot_stmt, rows)
    db.session.commit()
    current_app.logger.info(
        f"[SEQUENCE BACKFILL] membrane_proteins updated: {protein_result.rowcount}, "
        f"membrane_protein_uniprot updated: {uniprot_result.rowcount}."
    )

    protein_count = max(0, int(protein_result.rowcount or 0))
    uniprot_count = max(0, int(uniprot_result.rowcount or 0))
    _emit_progress(
        progress_callback,
        (
            "Backfilled missing sequences from TMAlphaFold for "
            f"{protein_count} membrane_proteins row(s) and {uniprot_count} membrane_protein_uniprot row(s)."
        ),
    )
    return {
        "protein_rows_updated": protein_count,
        "uniprot_rows_updated": uniprot_count,
        "attempted_targets": len(rows),
    }


def sync_tmalphafold_predictions(
    methods=None,
    with_tmdet=False,
    pdb_codes=None,
    limit=None,
    refresh=False,
    retry_errors=False,
    max_workers=8,
    timeout=30,
    backfill_sequences=True,
    progress_callback=None,
):
    _ensure_tmalphafold_storage(progress_callback=progress_callback)
    selected_methods = normalize_tmalphafold_methods(methods)
    targets = load_tmalphafold_targets(pdb_codes=pdb_codes, limit=limit)
    if not targets:
        summary = {
            "target_count": 0,
            "processed_predictions": 0,
            "methods": selected_methods,
            "with_tmdet": with_tmdet,
        }
        _emit_progress(progress_callback, "No eligible UniProt-backed records found for TMAlphaFold sync.")
        return summary

    _emit_progress(
        progress_callback,
        f"Loaded {len(targets)} UniProt-backed protein target(s) for TMAlphaFold sync.",
    )
    target_pairs = {
        (str(item["pdb_code"]).strip().upper(), str(item["uniprot_id"]).strip().upper())
        for item in targets
    }

    if not refresh:
        existing_query = TMAlphaFoldPrediction.query.with_entities(
            TMAlphaFoldPrediction.pdb_code,
            TMAlphaFoldPrediction.uniprot_id,
            TMAlphaFoldPrediction.method,
        ).filter(
            TMAlphaFoldPrediction.provider == "TMAlphaFold",
            TMAlphaFoldPrediction.method.in_(selected_methods + (["TMDET"] if with_tmdet else [])),
        )
        if pdb_codes:
            existing_query = existing_query.filter(
                db.func.upper(db.func.trim(TMAlphaFoldPrediction.pdb_code)).in_(
                    _normalize_selection(pdb_codes)
                )
            )
        if retry_errors:
            existing_query = existing_query.filter(
                TMAlphaFoldPrediction.status == "success"
            )
        existing = {
            (
                str(row.pdb_code or "").strip().upper(),
                str(row.uniprot_id or "").strip().upper(),
                str(row.method or "").strip(),
            )
            for row in existing_query
            if (
                str(row.pdb_code or "").strip().upper(),
                str(row.uniprot_id or "").strip().upper(),
            ) in target_pairs
        }
    else:
        existing = set()
    skipped_existing_jobs = 0

    existing_sequence_candidates = {
        "rows": [],
        "conflicting_targets": [],
    }
    if backfill_sequences:
        existing_rows_query = TMAlphaFoldPrediction.query.filter(
            TMAlphaFoldPrediction.provider == "TMAlphaFold",
            TMAlphaFoldPrediction.status == "success",
        )
        if target_pairs:
            existing_rows = [
                row
                for row in existing_rows_query.all()
                if (
                    str(row.pdb_code or "").strip().upper(),
                    str(row.uniprot_id or "").strip().upper(),
                ) in target_pairs
            ]
        else:
            existing_rows = []
        existing_sequence_candidates = _collect_sequence_backfills_from_stored_rows(
            existing_rows
        )

    client = TMAlphaFoldPredictorClient(timeout=timeout)
    futures = []
    predictions = []
    persist_batch_size = 25
    pending_upserts = []
    total_persisted_rows = 0
    total_inserted_rows = 0
    total_updated_rows = 0

    def flush_pending_predictions():
        nonlocal pending_upserts, total_persisted_rows, total_inserted_rows, total_updated_rows
        if not pending_upserts:
            current_app.logger.info(f"[FLUSH] Flushing batch of {len(pending_upserts)} prediction(s).")
            return {"stored_rows": 0, "inserted_rows": 0, "updated_rows": 0}
        stored_stats = _upsert_predictions(pending_upserts)
        pending_upserts = []
        total_persisted_rows += stored_stats["stored_rows"]
        total_inserted_rows += stored_stats["inserted_rows"]
        total_updated_rows += stored_stats["updated_rows"]
        return stored_stats

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        future_metadata = {}
        for target in targets:
            for method in selected_methods:
                key = (target["pdb_code"], target["uniprot_id"], method)
                if key in existing:
                    skipped_existing_jobs += 1
                    continue
                future = executor.submit(
                    client.fetch_method,
                    target["pdb_code"],
                    target["uniprot_id"],
                    method,
                )
                futures.append(future)
                future_metadata[future] = {
                    "pdb_code": target["pdb_code"],
                    "uniprot_id": target["uniprot_id"],
                    "method": method,
                }
            if with_tmdet:
                key = (target["pdb_code"], target["uniprot_id"], "TMDET")
                if key not in existing:
                    future = executor.submit(
                        client.fetch_tmdet,
                        target["pdb_code"],
                        target["uniprot_id"],
                    )
                    futures.append(future)
                    future_metadata[future] = {
                        "pdb_code": target["pdb_code"],
                        "uniprot_id": target["uniprot_id"],
                        "method": "TMDET",
                    }
                else:
                    skipped_existing_jobs += 1

        total_jobs = len(futures)
        _emit_progress(
            progress_callback,
            (
                f"Submitting {total_jobs} TMAlphaFold request(s) across {len(targets)} target(s). "
                f"Skipped {skipped_existing_jobs} already stored request(s)."
            ),
        )

        completed = 0
        for future in as_completed(futures):
            metadata = future_metadata.get(future, {})
            try:
                result = future.result()
            except Exception as exc:
                failed_method = str(metadata.get("method") or "").strip() or "unknown"
                result = TMAlphaFoldPredictionResult(
                    pdb_code=str(metadata.get("pdb_code") or "").strip().upper(),
                    uniprot_id=str(metadata.get("uniprot_id") or "").strip().upper(),
                    provider=TMALPHAFOLD_PROVIDER,
                    method=failed_method,
                    prediction_kind=(
                        "structure_membrane_plane"
                        if failed_method == "TMDET"
                        else "sequence_topology"
                    ),
                    tm_count=None,
                    tm_regions_json="[]",
                    raw_payload_json="",
                    source_url="",
                    status="error",
                    error_message=str(exc),
                    sequence_sequence=None,
                )
            predictions.append(result)
            pending_upserts.append(result)
            completed += 1
            if completed % persist_batch_size == 0 or completed == total_jobs:
                stored_batch = flush_pending_predictions()
                _emit_progress(
                    progress_callback,
                    (
                        f"Completed {completed}/{total_jobs} TMAlphaFold request(s). "
                        f"Persisted {stored_batch['stored_rows']} normalized prediction row(s) in this batch "
                        f"({stored_batch['inserted_rows']} inserted, {stored_batch['updated_rows']} updated; "
                        f"{total_persisted_rows} cumulative row(s) written so far)."
                    ),
                )

    updated_count = len(predictions)
    sequence_summary = {
        "protein_rows_updated": 0,
        "uniprot_rows_updated": 0,
        "attempted_targets": 0,
        "conflicting_targets": [],
        "enabled": bool(backfill_sequences),
    }
    if backfill_sequences:
        sequence_backfills = _collect_sequence_backfills(predictions)
        combined_sequence_rows = {
            (item["pdb_code"], item["uniprot_id"]): item
            for item in existing_sequence_candidates["rows"]
        }
        for item in sequence_backfills["rows"]:
            combined_sequence_rows[(item["pdb_code"], item["uniprot_id"])] = item
        sequence_summary.update(
            _persist_missing_sequences(
                combined_sequence_rows.values(),
                progress_callback=progress_callback,
            )
        )
        sequence_summary["conflicting_targets"] = sorted(
            {
                *existing_sequence_candidates["conflicting_targets"],
                *sequence_backfills["conflicting_targets"],
            }
        )
        if sequence_summary["conflicting_targets"]:
            _emit_progress(
                progress_callback,
                (
                    "Skipped TMAlphaFold sequence backfill for "
                    f"{len(sequence_summary['conflicting_targets'])} target(s) with inconsistent API sequences."
                ),
            )

    success_count = sum(1 for item in predictions if item.status == "success")
    error_count = sum(1 for item in predictions if item.status != "success")
    method_statuses = [
        {
            "method": item.method,
            "uniprot_id": item.uniprot_id,
            "status": item.status,
            "tm_count": item.tm_count,
            "tm_regions_json": item.tm_regions_json,
            "tm_regions": json.loads(item.tm_regions_json or "[]") if item.tm_regions_json else [],
            "source_url": item.source_url,
            "error_message": item.error_message,
        }
        for item in sorted(predictions, key=lambda value: (value.uniprot_id, value.method))
    ]

    summary = {
        "target_count": len(targets),
        "processed_predictions": updated_count,
        "persisted_rows": total_persisted_rows,
        "inserted_rows": total_inserted_rows,
        "updated_rows": total_updated_rows,
        "skipped_existing_jobs": skipped_existing_jobs,
        "successful_predictions": success_count,
        "failed_predictions": error_count,
        "methods": selected_methods,
        "with_tmdet": with_tmdet,
        "retry_errors": bool(retry_errors),
        "max_workers": max_workers,
        "refresh": bool(refresh),
        "sequence_backfill": sequence_summary,
        "method_statuses": method_statuses,
    }
    _emit_progress(
        progress_callback,
        f"TMAlphaFold sync stored {success_count} successful and {error_count} failed prediction record(s).",
    )
    return summary


def get_normalized_tm_prediction_summaries(pdb_code, providers=None):
    mapping = get_normalized_tm_prediction_summaries_for_pdb_codes(
        [pdb_code],
        providers=providers,
    )
    normalized_code = str(pdb_code or "").strip().upper()
    return mapping.get(normalized_code, [])


def get_normalized_tm_prediction_summaries_for_pdb_codes(pdb_codes, providers=None):
    normalized_codes = [
        str(code or "").strip().upper()
        for code in pdb_codes or ()
        if str(code or "").strip()
    ]
    normalized_codes = list(dict.fromkeys(normalized_codes))
    if not normalized_codes:
        return {}

    query = TMAlphaFoldPrediction.query.filter(
        db.func.upper(db.func.trim(TMAlphaFoldPrediction.pdb_code)).in_(normalized_codes),
        TMAlphaFoldPrediction.status == "success",
    )
    normalized_providers = [
        str(provider or "").strip()
        for provider in providers or ()
        if str(provider or "").strip()
    ]
    if normalized_providers:
        query = query.filter(TMAlphaFoldPrediction.provider.in_(normalized_providers))
    rows = query.all()

    grouped = defaultdict(list)
    for row in rows:
        normalized_pdb_code = str(row.pdb_code or "").strip().upper()
        grouped[(normalized_pdb_code, row.provider, row.method, row.prediction_kind)].append(row)

    summaries_by_code = defaultdict(list)
    for (normalized_pdb_code, provider, method, prediction_kind), method_rows in grouped.items():
        tm_count_values = {row.tm_count for row in method_rows if row.tm_count is not None}
        region_values = {
            str(row.tm_regions_json or "[]")
            for row in method_rows
        }
        consensus = len(tm_count_values) <= 1 and len(region_values) <= 1
        sample = method_rows[0]
        summaries_by_code[normalized_pdb_code].append(
            {
                "provider": provider,
                "method": method,
                "prediction_kind": prediction_kind,
                "pdb_code": normalized_pdb_code,
                "uniprot_ids": sorted({row.uniprot_id for row in method_rows if row.uniprot_id}),
                "accession_count": len({row.uniprot_id for row in method_rows if row.uniprot_id}),
                "tm_count": next(iter(tm_count_values)) if consensus and tm_count_values else None,
                "tm_regions_json": sample.tm_regions_json if consensus else "[]",
                "source_urls": [row.source_url for row in method_rows if row.source_url],
                "consensus": consensus,
                "ambiguous": not consensus,
                "note": (
                    "Multiple UniProt-backed results disagree for this PDB entry."
                    if not consensus
                    else None
                ),
            }
        )
    for code in summaries_by_code:
        summaries_by_code[code].sort(
            key=lambda item: (item["provider"], item["prediction_kind"], item["method"])
        )
    return dict(summaries_by_code)


def get_tmalphafold_prediction_summaries(pdb_code):
    return get_normalized_tm_prediction_summaries(pdb_code, providers=["TMAlphaFold"])
