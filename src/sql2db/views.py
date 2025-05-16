
from flask import request
from flask_restful import Resource
from src.sql2db.sql_builder import nl_to_sql_multi
from src.utils.response import ApiResponse
from src.sql2db.executor import execute_sql

class SqlGenerator(Resource):
    def post(self):
        data = request.get_json(force=True)
        payload = {
            "schemas": [
                "Table membrane_proteins(group, subgroup, pdb_code, is_master_protein, name, species, "
                "taxonomic_domain, expressed_in_species, resolution)",

                "Table membrane_protein_uniprot(uniprot_id, pdb_code, info_type, info_created, "
                "organism_scientific_name, organism_common_name, organism_lineage, protein_recommended_name, "
                "protein_alternative_name, associated_genes, comment_function, comment_interactions, "
                "comment_catalytic_activity, comment_subunit, comment_PTM, comment_caution, "
                "comment_subcellular_locations, comment_alternative_products, comment_disease_name, "
                "comment_disease, comment_similarity, features, references, keywords, sequence_length, "
                "sequence_mass, extra_attributes, sequence_sequence, molecular_function, cellular_component, "
                "biological_process, cross_references)",

                "Table membrane_protein_opm(name, description, comments, resolution, topology_subunit, "
                "topology_show_in, thickness, thicknesserror, subunit_segments, tilt, tilterror, gibbs, tau, "
                "verification, family_name_cache, species_name_cache, membrane_name_cache, "
                "secondary_representations_count, structure_subunits_count, citations_count, uniprotcodes, "
                "subunits, secondary_representations, citations, family_name, family_pfam, family_interpro, "
                "family_tcdb, family_primary_structures_count, family_superfamily_name, family_superfamily_pfam, "
                "family_superfamily_tcdb, family_superfamily_families_count, "
                "family_superfamily_classtype_name, family_superfamily_classtype_superfamilies_count, "
                "family_superfamily_classtype_type_name, "
                "family_superfamily_classtype_type_classtypes_count, species_name, species_description, "
                "species_primary_structures_count, membrane_name, membrane_primary_structures_count, "
                "membrane_short_name, membrane_abbrevation, membrane_topology_in, membrane_topology_out, "
                "membrane_lipid_references, membrane_lipid_pubmed, pdb_code)"
            ],
            "query": data.get('query')
        }
        schemas = payload.get('schemas')
        nlq = payload.get('query')
        if not schemas or not nlq:
            return {"error": "Missing 'schemas' or 'query' in payload."}, 400

        try:
            sql = nl_to_sql_multi(nlq, schemas)
            rows  = execute_sql(sql)
        except Exception as e:
            return ApiResponse.error(message=f"An error occurred {e}", status_code=400)

        return ApiResponse.success(
            message="SQL generated successfully",
            data={
                "sql": sql,
                "rows": rows
            }, status_code=200
        )
