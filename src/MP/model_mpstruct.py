from datetime import datetime
from database.db import db

class MPSTURC(db.Model):
    __tablename__ = 'membrane_protein_mpstruct'
    id = db.Column(db.Integer, primary_key=True)
    group = db.Column(db.Text)
    subgroup = db.Column(db.Text)
    pdb_code = db.Column(db.Text)
    is_master_protein = db.Column(db.Text)
    name = db.Column(db.Text)
    species = db.Column(db.Text)
    taxonomic_domain = db.Column(db.Text)
    expressed_in_species = db.Column(db.Text)
    resolution = db.Column(db.Text)
    description = db.Column(db.Text)
    related_pdb_entries = db.Column(db.Text)
