from datetime import datetime

from database.db import db


class TMAlphaFoldPrediction(db.Model):
    __tablename__ = "membrane_protein_tmalphafold_predictions"
    __table_args__ = (
        db.UniqueConstraint(
            "provider",
            "method",
            "pdb_code",
            "uniprot_id",
            name="uq_tmalphafold_provider_method_pdb_uniprot",
        ),
    )

    id = db.Column(db.Integer, primary_key=True)
    pdb_code = db.Column(db.Text, nullable=False, index=True)
    uniprot_id = db.Column(db.Text, nullable=False, index=True)
    provider = db.Column(db.Text, nullable=False, default="TMAlphaFold", index=True)
    method = db.Column(db.Text, nullable=False, index=True)
    prediction_kind = db.Column(db.Text, nullable=False, default="sequence_topology")
    tm_count = db.Column(db.Integer)
    tm_regions_json = db.Column(db.Text)
    raw_payload_json = db.Column(db.Text)
    source_url = db.Column(db.Text)
    status = db.Column(db.Text, nullable=False, default="success", index=True)
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
