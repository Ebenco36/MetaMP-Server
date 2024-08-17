"""
from datetime import datetime
from database.db import db

class TableOPM(db.Model):
    __tablename__ = 'tables_opm'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    types = db.Column(db.Integer)
    classtypes = db.Column(db.Integer)
    superfamilies = db.Column(db.Integer)
    family = db.Column(db.Integer)
    primary_structure = db.Column(db.Integer)
    species = db.Column(db.Integer)
    membrane = db.Column(db.Integer)
    assembly = db.Column(db.Integer)
    assembly_families = db.Column(db.Integer)
    assembly_membrane = db.Column(db.Integer)
    assembly_superfamilies = db.Column(db.Integer)

    parent_id = db.Column(db.Integer, db.ForeignKey('tables_opm.id'))
    parent = db.relationship('TableOPM', remote_side=[id])

    structure_stats = db.relationship('StructureStatsOPM', backref='table', lazy=True)

class StructureStatsOPM(db.Model):
    __tablename__ = 'structure_stats_opm'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    primary_structures_count = db.Column(db.Integer)
    secondary_representations_count = db.Column(db.Integer)
    table_id = db.Column(db.Integer, db.ForeignKey('tables_opm.id'), nullable=False)

    classes = db.relationship('ClassOPM', backref='structure_stats_opm', lazy=True)

class ClassOPM(db.Model):
    __tablename__ = 'classes_opm'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    primary_structures_count = db.Column(db.Integer)
    secondary_representations_count = db.Column(db.Integer)
    structure_stats_id = db.Column(db.Integer, db.ForeignKey('structure_stats_opm.id'), nullable=False)
    
    
class SuperfamiliesOPM(db.Model):
    __tablename__ = 'superfamilies_opm'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    pfam = db.Column(db.String(255))
    tcdb = db.Column(db.String(255))
    families_count = db.Column(db.Integer)

class FamilyOPM(db.Model):
    __tablename__ = 'family_opm'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    pfam = db.Column(db.String(20))
    interpro = db.Column(db.String(20))
    tcdb = db.Column(db.String(20))
    primary_structures_count = db.Column(db.Integer)
    
    # Define relationships with other tables
    superfamily_id = db.Column(db.Integer, db.ForeignKey('superfamilies_opm.id'))


class SpeciesOPM(db.Model):
    __tablename__ = 'species_opm'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    description = db.Column(db.String(255))
    primary_structures_count = db.Column(db.Integer)



class MembraneOPM(db.Model):
    __tablename__ = 'membrane_opm'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    primary_structures_count = db.Column(db.Integer)
    short_name = db.Column(db.String(50))
    abbrevation = db.Column(db.String(50))
    topology_in = db.Column(db.String(255))
    topology_out = db.Column(db.String(255))
    lipid_references = db.Column(db.String(255))
    lipid_pubmed = db.Column(db.String(255))
    
    
class UniprotCodeOPM(db.Model):
    __tablename__ = 'uniprot_codes_opm'

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20))
    membrane_protein_id = db.Column(db.Integer, db.ForeignKey('membrane_proteins_opm.id'), nullable=False)

 
class MembraneProteinOPM(db.Model):
    __tablename__ = 'membrane_proteins_opm'

    id = db.Column(db.Integer, primary_key=True)
    ordering = db.Column(db.Integer)
    pdbid = db.Column(db.String(10))
    name = db.Column(db.String(255))
    description = db.Column(db.String(255))
    comments = db.Column(db.String(255))
    resolution = db.Column(db.String(10))
    topology_subunit = db.Column(db.String(1))
    topology_show_in = db.Column(db.Boolean)
    thickness = db.Column(db.Float)
    thicknesserror = db.Column(db.Float)
    subunit_segments = db.Column(db.Integer)
    tilt = db.Column(db.Float)
    tilterror = db.Column(db.Float)
    gibbs = db.Column(db.Float)
    tau = db.Column(db.String(255))
    verification = db.Column(db.String(255))
    family_name_cache = db.Column(db.String(255))
    species_name_cache = db.Column(db.String(255))
    membrane_name_cache = db.Column(db.String(255))
    # Define relationships with other tables
    membrane_id = db.Column(db.Integer, db.ForeignKey('membrane_opm.id'))
    species_id = db.Column(db.Integer, db.ForeignKey('species_opm.id'))
    family_id = db.Column(db.Integer, db.ForeignKey('family_opm.id'))
    superfamily_id = db.Column(db.Integer, db.ForeignKey('superfamilies_opm.id'))
    classtype_id = db.Column(db.Integer, db.ForeignKey('classes_opm.id'))
    type_id = db.Column(db.Integer, db.ForeignKey('structure_stats_opm.id'))
    
    membrane = db.relationship('MembraneOPM', backref='membrane_proteins')
    species = db.relationship('SpeciesOPM', backref='membrane_proteins')
    family = db.relationship('FamilyOPM', backref='membrane_proteins')
    superfamily = db.relationship('SuperfamiliesOPM', backref='membrane_proteins')
    classtype = db.relationship('ClassOPM', backref='membrane_proteins')
    structure_stats = db.relationship('StructureStatsOPM', backref='membrane_proteins')

    secondary_representations_count = db.Column(db.Integer)
    structure_subunits_count = db.Column(db.Integer)
    citations_count = db.Column(db.Integer)
    
    # Define a relationship with UniprotCodes model
    uniprotcodes = db.relationship('UniprotCodeOPM', backref='membrane_protein_opm', lazy=True)
"""