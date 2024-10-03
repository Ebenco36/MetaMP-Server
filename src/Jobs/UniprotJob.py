# import pandas as pd
# import requests
# import os

# data_path = os.environ.get("AIRFLOW_HOME")
# if (data_path):
#     modified_path = data_path.replace("/airflow_home", "")
# else:
#     modified_path = "."

# class UniProtDataFetcher:
#     def __init__(self):
#         self.base_url = "https://www.ebi.ac.uk/proteins/api/proteins/"
#         self.uniparc_base_url = "https://rest.uniprot.org/uniparc/search"

#     def fetch_data(self, uniprot_id):
#         url = self.base_url + str(uniprot_id)
#         response = requests.get(url)
#         if response.status_code == 200:
#             return "uniprotKB", response.json()
#         else:
#             print(f"Failed to retrieve data for UniProt ID: {uniprot_id}")
#             return "UniParc", self.fetch_from_uniparc(uniprot_id)

#     def fetch_from_uniparc(self, uniprot_id):
#         query_url = f"{self.uniparc_base_url}?query={uniprot_id}&format=json"
#         response = requests.get(query_url)

#         if response.status_code == 200:
#             retrieved_data = response.json()
#             # get the first item from the list. Just extract basic information
#             return retrieved_data.get("results", {})[0]
#         else:
#             print(f"Failed to retrieve data for UniProt ID: {uniprot_id} from UniParc.")
#             return None
        
#     def extract_information(self, uniprot_data, data_source="uniprotKB"):
#         data_format = {
#             "uniprot_id": "",
#             "info_type": "",
#             "info_created": "",
#             "info_modified": "",
#             "organism_scientific_name": "",
#             "organism_common_name": "",
#             "organism_lineage": "",
#             "secondary_accession": "",
#             "protein_recommended_name": "",
#             "protein_alternative_name": "",
#             "associated_genes": "",
#             "comment_function": [],
#             "comment_interactions": [],
#             "comment_subcellular_locations": [],
#             "comment_alternative_products": [],
#             "comment_disease_name": "",
#             "comment_disease": [],
#             "comment_similarity": [],
#             "features": [],
#             "references": [],
#             "keywords": [],
#             "sequence_length": 0,
#             "sequence_mass": 0,
#             "sequence_sequence": "",
#             "molecular_function": "",
#             "cellular_component": "",
#             "biological_process": ""
            
#         }
#         molecular_function = ""
#         cellular_component = ""
#         biological_process = ""
        
#         if data_source == "uniprotKB":
#             if "accession" in uniprot_data:
#                 data_format["uniprot_id"] = uniprot_data['accession']
                
#             if "info" in uniprot_data:
#                 info = uniprot_data['info']
#                 data_format["info_type"] = info["type"]
#                 data_format["info_created"] = info["created"]
#                 data_format["info_modified"] = info["modified"]
            
#             if "secondaryAccession" in uniprot_data:
#                 secondary_accession = uniprot_data['secondaryAccession']
#                 data_format["secondary_accession"] = "; ".join(secondary_accession)
                
#             if "organism" in uniprot_data:
#                 organism = uniprot_data['organism']
#                 data_format["organism_scientific_name"] = next((name["value"] for name in organism.get("names", []) if name.get("type") == "scientific"), None)
#                 data_format["organism_common_name"] = next((name["value"] for name in organism.get("names", []) if name.get("type") == "common"), None)
#                 data_format["organism_lineage"] = "; ".join(organism.get("lineage", []))
            
            
#             if "protein" in uniprot_data:
#                 protein = uniprot_data['protein']
#                 data_format["protein_recommended_name"] = next((name.get("fullName")["value"] for name in organism.get("recommendedName", []) if name.get("fullName") != ""), None)
#                 data_format["protein_alternative_name"] = next((name.get("fullName")["value"] for name in organism.get("alternativeName", []) if name.get("fullName") != ""), None)
                
#             if "gene" in uniprot_data:
#                 genes = uniprot_data['gene']
#                 values = [item["name"]["value"] for item in genes if "name" in item and "value" in item["name"]]
#                 data_format["associated_genes"] = "; ".join(values)
                
#             if "comments" in uniprot_data:
#                 comments = uniprot_data["comments"]
#                 for comment in comments:
#                     comment_type = comment.get("type", "")
#                     if comment_type == "FUNCTION":
#                         data_format["comment_function"] = comment.get("text", "")
#                     if comment_type == "INTERACTION":
#                         data_format["comment_interactions"] = comment.get("interactions", "")
#                     if comment_type == "SUBCELLULAR_LOCATION":
#                         data_format["comment_subcellular_locations"] = comment.get("locations", "")
#                     if comment_type == "ALTERNATIVE_PRODUCTS":
#                         data_format["comment_alternative_products"] = comment.get("isoforms", "")
#                     if comment_type == "DISEASE":
#                         data_format["comment_disease_name"] = comment.get("diseaseId", "")
#                         data_format["comment_disease"] = comment
#                     if comment_type == "SIMILARITY":
#                         data_format["comment_similarity"] = comment.get("text", "")
                        
#             if "features" in uniprot_data:
#                 features = uniprot_data["features"]
#                 data_format["features"] = features
            
#             if "references" in uniprot_data:
#                 references = uniprot_data["references"]
#                 data_format["references"] = references
            
#             if "keywords" in uniprot_data:
#                 keywords = uniprot_data["keywords"]
#                 data_format["keywords"] = keywords
                
#             if "sequence" in uniprot_data:
#                 sequence = uniprot_data["sequence"]
#                 data_format["sequence_length"] = sequence.get("length", 0)
#                 data_format["sequence_mass"] = sequence.get("mass", 0)
#                 data_format["sequence_sequence"] = sequence.get("sequence", 0)
                
#             if 'dbReferences' in uniprot_data:
#                 for db_ref in uniprot_data['dbReferences']:
#                     if 'properties' in db_ref:
#                         if 'term' in db_ref['properties']:
#                             term_type = db_ref['properties']['term'][0]
#                             term_content = db_ref['properties']['term'][2:]
#                             if term_type == 'F':
#                                 molecular_function += term_content + "; "
#                             elif term_type == 'C':
#                                 cellular_component += term_content + "; "
#                             elif term_type == 'P':
#                                 biological_process += term_content + "; "
                                
#             data_format["molecular_function"] = molecular_function.strip("; ")
#             data_format["cellular_component"] = cellular_component.strip("; ")
#             data_format["biological_process"] = biological_process.strip("; ")
#         else:
#             # For UniParc
#             if "uniParcCrossReferences" in uniprot_data:
#                 info_id = uniprot_data['uniParcCrossReferences'][0]
#                 data_format["uniprot_id"] = info_id['id']
                
#             if "uniParcId" in uniprot_data:
#                 data_format["info_type"] = "uniprot_data: " + uniprot_data["uniParcId"]
                
#             if "uniParcCrossReferences" in uniprot_data:
#                 info = uniprot_data['uniParcCrossReferences'][-1]
#                 data_format["info_created"] = info["created"]
#                 data_format["info_modified"] = info["lastUpdated"]
                
#                 if "geneName" in info:
#                     gene_name = info["geneName"]
#                     data_format["associated_genes"] = gene_name
                
#                 if "organism" in info:
#                     organism = info["organism"]
#                     data_format["organism_scientific_name"] = organism["scientificName"]
                    
#             if "sequence" in uniprot_data:
#                 sequence = uniprot_data["sequence"]
#                 data_format["sequence_length"] = sequence.get("length", 0)
#                 data_format["sequence_mass"] = sequence.get("molWeight", 0)
#                 data_format["sequence_sequence"] = sequence.get("value", 0)
                
#             if "features" in uniprot_data:
#                 features = uniprot_data["sequenceFeatures"]
#                 data_format["features"] = features
       
#         return data_format


#     def fetch_and_process(self, df, uniprot_id_column, function_file):
#         # Load existing function data
#         try:
#             function_data = pd.read_csv(function_file)
#         except FileNotFoundError:
#             function_data = pd.DataFrame(
#                 columns=[
#                     'UniProt_ID', 
#                     'pdb_code',
#                     "info_type",
#                     "info_created",
#                     "info_modified",
#                     "organism_scientific_name",
#                     "organism_common_name",
#                     "organism_lineage",
#                     "secondary_accession",
#                     "protein_recommended_name",
#                     "protein_alternative_name",
#                     "associated_genes",
#                     "comment_function",
#                     "comment_interactions",
#                     "comment_subcellular_locations",
#                     "comment_alternative_products",
#                     "comment_disease_name",
#                     "comment_disease",
#                     "comment_similarity",
#                     "features",
#                     "references",
#                     "keywords",
#                     "sequence_length",
#                     "sequence_mass",
#                     "sequence_sequence",
#                     'Molecular_Function', 
#                     'Cellular_Component', 
#                     'Biological_Process'
#                 ]
#             )
        
#         # Initialize data dictionary
#         data = {
#             'UniProt_ID': [], 
#             'pdb_code': [],
#             'data_source': [],
#             "info_type": [],
#             "info_created": [],
#             "info_modified": [],
#             "organism_scientific_name": [],
#             "organism_common_name": [],
#             "organism_lineage": [],
#             "secondary_accession": [],
#             "protein_recommended_name": [],
#             "protein_alternative_name": [],
#             "associated_genes": [],
#             "comment_function": [],
#             "comment_interactions": [],
#             "comment_subcellular_locations": [],
#             "comment_alternative_products": [],
#             "comment_disease_name": [],
#             "comment_disease": [],
#             "comment_similarity": [],
#             "features": [],
#             "references": [],
#             "keywords": [],
#             "sequence_length": [],
#             "sequence_mass": [],
#             "sequence_sequence": [],
#             'Molecular_Function': [], 
#             'Cellular_Component': [], 
#             'Biological_Process': []
#         }
        
#         # Iterate over each row in the DataFrame
#         for index, row in df.iterrows():
#             uniprot_id = row[uniprot_id_column]
#             pdb_code = row["Pdb Code"]
#             # Check if the entry already exists in the function file
#             if uniprot_id in function_data['UniProt_ID'].values:
#                 # If entry already exists, fetch data from function file
#                 molecular_function = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'Molecular_Function'].values[0]
#                 cellular_component = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'Cellular_Component'].values[0]
#                 biological_process = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'Biological_Process'].values[0]
#                 pdb_code = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'pdb_code'].values[0]
#                 data_source = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'data_source'].values[0]
#                 features = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'features'].values[0]
#                 keywords = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'keywords'].values[0]
#                 info_type = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'info_type'].values[0]
#                 references = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'references'].values[0]
#                 info_created = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'info_created'].values[0]
#                 sequence_mass = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'sequence_mass'].values[0]
#                 info_modified = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'info_modified'].values[0]
#                 sequence_length = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'sequence_length'].values[0]
#                 comment_disease = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'comment_disease'].values[0]
#                 organism_lineage = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'organism_lineage'].values[0]
#                 associated_genes = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'associated_genes'].values[0]
#                 comment_function = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'comment_function'].values[0]
#                 sequence_sequence = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'sequence_sequence'].values[0]
#                 comment_disease_name = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'comment_disease_name'].values[0]
#                 comment_similarity = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'comment_similarity'].values[0]
#                 secondary_accession = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'secondary_accession'].values[0]
#                 organism_common_name = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'organism_common_name'].values[0]
#                 comment_interactions = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'comment_interactions'].values[0]
                
                
#                 organism_scientific_name = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'organism_scientific_name'].values[0]
#                 protein_recommended_name = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'protein_recommended_name'].values[0]
#                 protein_alternative_name = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'protein_alternative_name'].values[0]
#                 comment_subcellular_locations = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'comment_subcellular_locations'].values[0]
#                 comment_alternative_products = function_data.loc[function_data['UniProt_ID'] == uniprot_id, 'comment_alternative_products'].values[0]
#             else:
#                 # If entry does not exist, fetch data from UniProt
#                 data_source, uniprot_data = self.fetch_data(uniprot_id)
                
#                 if uniprot_data:
#                     extracted_data = self.extract_information(uniprot_data, data_source)

#                     features = extracted_data["features"]
#                     keywords = extracted_data["keywords"]
#                     info_type = extracted_data["info_type"]
#                     uniprot_id = extracted_data["uniprot_id"]
#                     references = extracted_data["references"]
#                     info_created = extracted_data["info_created"]
#                     sequence_mass = extracted_data["sequence_mass"]
#                     info_modified = extracted_data["info_modified"]
#                     sequence_length = extracted_data["sequence_length"]
#                     comment_disease = extracted_data["comment_disease"]
#                     organism_lineage = extracted_data["organism_lineage"]
#                     associated_genes = extracted_data["associated_genes"]
#                     comment_function = extracted_data["comment_function"]
#                     sequence_sequence = extracted_data["sequence_sequence"]
#                     comment_disease_name = extracted_data["comment_disease_name"]
#                     comment_similarity = extracted_data["comment_similarity"]
#                     molecular_function = extracted_data["molecular_function"]
#                     cellular_component = extracted_data["cellular_component"]
#                     biological_process = extracted_data["biological_process"]
#                     secondary_accession = extracted_data["secondary_accession"]
#                     organism_common_name = extracted_data["organism_common_name"]
#                     comment_interactions = extracted_data["comment_interactions"]
#                     organism_scientific_name = extracted_data["organism_scientific_name"]
#                     protein_recommended_name = extracted_data["protein_recommended_name"]
#                     protein_alternative_name = extracted_data["protein_alternative_name"]
#                     comment_subcellular_locations = extracted_data["comment_subcellular_locations"]
#                     comment_alternative_products = extracted_data["comment_alternative_products"]
                    
#                     data['UniProt_ID'].append(uniprot_id)
#                     data['pdb_code'].append(pdb_code)
#                     data['data_source'].append(data_source)
#                     data['Molecular_Function'].append(molecular_function)
#                     data['Cellular_Component'].append(cellular_component)
#                     data['Biological_Process'].append(biological_process)
#                     data['info_type'].append(info_type)
#                     data['info_created'].append(info_created)
#                     data['info_modified'].append(info_modified)
#                     data['keywords'].append(keywords)
#                     data['organism_scientific_name'].append(organism_scientific_name)
#                     data['organism_common_name'].append(organism_common_name)
#                     data['protein_recommended_name'].append(protein_recommended_name)
#                     data['protein_alternative_name'].append(protein_alternative_name)
#                     data['secondary_accession'].append(secondary_accession)
#                     data['comment_disease_name'].append(comment_disease_name)
#                     data['sequence_length'].append(sequence_length)
#                     data['sequence_mass'].append(sequence_mass)
#                     data['sequence_sequence'].append(sequence_sequence)
#                     data['organism_lineage'].append(organism_lineage)
#                     data['associated_genes'].append(associated_genes)
#                     data['references'].append(references)
#                     data['features'].append(features)
#                     data['comment_disease'].append(comment_disease)
#                     data['comment_function'].append(comment_function)
#                     data['comment_similarity'].append(comment_similarity)
#                     data['comment_interactions'].append(comment_interactions)
#                     data['comment_subcellular_locations'].append(comment_subcellular_locations)
#                     data['comment_alternative_products'].append(comment_alternative_products)
                    
#         # Append data to function data DataFrame
#         new_function_data = pd.DataFrame(data)
#         function_data = pd.concat([function_data, new_function_data], ignore_index=True)
        
#         # Update function file
#         function_data.to_csv(function_file, index=False)
        
#         return function_data


# # import data
# data = pd.read_csv(modified_path + "/datasets/Quantitative_data.csv", low_memory=False)
# data = data[data["uniprot_id"].notna()]
# # data = data[data["uniprot_id"] == "Q8N6U8"]
# fetcher = UniProtDataFetcher()
# result_df = fetcher.fetch_and_process(data, 'uniprot_id', modified_path + '/datasets/Uniprot_functions.csv')



import pandas as pd
import requests
import os

data_path = os.environ.get("AIRFLOW_HOME", ".")
modified_path = data_path.replace("/airflow_home", "") if data_path else "."

class UniProtDataFetcher:
    def __init__(self):
        self.search_url = "https://rest.uniprot.org/uniprotkb/search"
        self.old_base_url = "https://www.ebi.ac.uk/proteins/api/proteins/"
        self.uniparc_base_url = "https://rest.uniprot.org/uniparc/search"
    
    def fetch_data(self, pdb_code):
        # Search for all records associated with the PDB code
        query_url = f"{self.search_url}?query=({pdb_code})&format=json"
        response = requests.get(query_url)
        if response.status_code == 200 and len(response.json().get("results", [])) > 0:
            return "uniprotKB", response.json().get("results", [])
        else:
            print(f"Failed to retrieve data for PDB code: {pdb_code}")
            return "UniParc", self.fetch_from_uniparc(pdb_code)

    def fetch_from_uniparc(self, pdb_code):
        query_url = f"{self.uniparc_base_url}?query={pdb_code}&format=json"
        response = requests.get(query_url)

        if response.status_code == 200 and len(response.json().get("results", [])) > 0:
            retrieved_data = response.json()
            # get the first item from the list. Just extract basic information
            return retrieved_data.get("results", {})[0]
        else:
            print(f"Failed to retrieve data for UniProt ID: {pdb_code} from UniParc.")
            return []
        
    def extract_information(self, uniprot_data, data_source="uniprotKB"):
        # Initialize empty data structure
        data_format = {
            "uniprot_id": "",
            "uniProtkb_id": "",
            "info_type": "",
            "info_created": "",
            "info_modified": "",
            "info_sequence_update": "",
            "annotation_score": "",
            "taxon_id": "",
            "organism_scientific_name": "",
            "organism_common_name": "",
            "organism_lineage": "",
            "secondary_accession": "",
            "protein_recommended_name": "",
            "protein_alternative_name": "",
            "associated_genes": "",
            "comment_function": [],
            "comment_interactions": [],
            "comment_catalytic_activity": [],
            "comment_subunit": [],
            "comment_PTM": [],
            "comment_caution": [],
            "comment_tissue_specificity": [],
            "comment_subcellular_locations": [],
            "comment_alternative_products": [],
            "comment_disease_name": "",
            "comment_disease": [],
            "comment_similarity": [],
            "features": [],
            "references": [],
            "cross_references": [],
            "keywords": [],
            "sequence_length": 0,
            "sequence_mass": 0,
            "extra_attributes": [],
            "sequence_sequence": "",
            "molecular_function": "",
            "cellular_component": "",
            "biological_process": ""
        }
        
        disease_list = []
        
        if data_source == "uniprotKB":
            
            if "primaryAccession" in uniprot_data:
                data_format["uniprot_id"] = uniprot_data.get("primaryAccession", "")
            if "uniProtkbId" in uniprot_data:
                data_format["uniProtkb_id"] = uniprot_data.get("uniProtkbId", "")
            if "entryType" in uniprot_data:
                data_format["info_type"] = uniprot_data.get("entryType", "")
            if "entryAudit" in uniprot_data:
                entry_audit = uniprot_data.get("entryAudit", {})
                data_format["info_created"] = entry_audit.get("firstPublicDate", "")
                data_format["info_modified"] = entry_audit.get("lastAnnotationUpdateDate", "")
                data_format["info_sequence_update"] = entry_audit.get("lastSequenceUpdateDate", "")
            if "annotationScore" in uniprot_data:
                data_format["annotation_score"] = uniprot_data.get("annotationScore", "")
            if "organism" in uniprot_data:
                organism = uniprot_data.get("organism", {})
                data_format["taxon_id"] = organism.get("taxonId", "")
                data_format["organism_scientific_name"] = organism.get("scientificName", "")
                data_format["organism_common_name"] = organism.get("commonName", "")
                data_format["organism_lineage"] = "; ".join(organism.get("lineage", []))
            if "secondaryAccessions" in uniprot_data:
                data_format["secondary_accession"] = "; ".join(uniprot_data.get("secondaryAccessions", []))
            if "proteinDescription" in uniprot_data:
                data_format["protein_recommended_name"] = uniprot_data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
                data_format["protein_alternative_name"] = "; ".join([name["fullName"]["value"] for name in uniprot_data.get("proteinDescription", {}).get("alternativeNames", []) if "fullName" in name])
            if "genes" in uniprot_data:
                data_format["associated_genes"] = "; ".join([gene["geneName"]["value"] for gene in uniprot_data.get("genes", []) if "geneName" in gene])
            
            # Extract detailed comments
            if "comments" in uniprot_data:
                comments = uniprot_data["comments"]
                for comment in comments:
                    comment_type = comment.get("commentType", "")
                    if comment_type == "FUNCTION":
                        comment_function = comment.get("texts", [])
                        data_format["comment_function"] = comment_function
                    if comment_type == "INTERACTION":
                        comment_interaction = comment.get("interactions", "")
                        data_format["comment_interactions"] = comment_interaction
                    if comment_type == "SUBCELLULAR LOCATION":
                        comment_subcellularLocations = comment.get("subcellularLocations", "")
                        data_format["comment_subcellular_locations"] = comment_subcellularLocations
                    if comment_type == "ALTERNATIVE PRODUCTS":
                        comment_isoforms = comment.get("isoforms", "")
                        data_format["comment_alternative_products"] = comment_isoforms
                    if comment_type == "DISEASE":
                        disease_info = comment.get("disease", {})
                        data_format["comment_disease"].append(disease_info)
                        disease_list.append(disease_info.get("diseaseId", ""))
                        
                    if comment_type == "TISSUE SPECIFICITY":
                        comment_tissue_specificity = comment.get("texts", [])
                        data_format["comment_tissue_specificity"] = comment_tissue_specificity
                        
                    if comment_type == "SUBUNIT":
                        comment_subunit = comment.get("texts", [])
                        data_format["comment_subunit"] = comment_subunit
                        
                    if comment_type == "PTM":
                        comment_PTM = comment.get("texts", [])
                        data_format["comment_PTM"] = comment_PTM
                    
                    if comment_type == "CATALYTIC ACTIVITY":  
                        comment_catalytic_activity = comment.get("reaction", {})
                        data_format["comment_catalytic_activity"].append(comment_catalytic_activity)
                        
                    if comment_type == "SIMILARITY":
                        comment_similarity = comment.get("texts", [])
                        data_format["comment_similarity"] = comment_similarity
                        
                    if comment_type == "CAUTION":
                        comment_caution = comment.get("texts", [])
                        data_format["comment_caution"] = comment_caution
                        
                data_format["comment_disease_name"] = "; ".join(disease_list)
            # Extract other detailed data
            if "features" in uniprot_data:
                data_format["features"] = uniprot_data.get("features", [])
                
            if "references" in uniprot_data:
                data_format["references"] = uniprot_data.get("references", [])
                
            if "keywords" in uniprot_data:
                data_format["keywords"] = uniprot_data.get("keywords", [])
                    
            if "sequence" in uniprot_data:
                sequence = uniprot_data.get("sequence", {})
                data_format["sequence_length"] = sequence.get("length", 0)
                data_format["sequence_mass"] = sequence.get("molWeight", 0)
                data_format["sequence_sequence"] = sequence.get("value", "")
                
            if "extraAttributes" in uniprot_data:
                data_format["extra_attributes"] = uniprot_data["extraAttributes"]
               
            if "uniProtKBCrossReferences" in uniprot_data:   
                for db_ref in uniprot_data.get("uniProtKBCrossReferences", []):
                    for prop in db_ref.get("properties", []):
                        term = prop.get("value", "")
                        if term.startswith('F:'):
                            data_format["molecular_function"] += term[2:] + "; "
                        elif term.startswith('C:'):
                            data_format["cellular_component"] += term[2:] + "; "
                        elif term.startswith('P:'):
                            data_format["biological_process"] += term[2:] + "; "
            
                   
                data_format["cross_references"] = uniprot_data.get("uniProtKBCrossReferences", [])
            
            # Strip trailing semicolons and spaces
            data_format["molecular_function"] = data_format["molecular_function"].strip("; ")
            data_format["cellular_component"] = data_format["cellular_component"].strip("; ")
            data_format["biological_process"] = data_format["biological_process"].strip("; ")
        else:
            # For UniParc
            if "uniParcCrossReferences" in uniprot_data:
                data_format["uniprot_id"] = "uniParcId: " + uniprot_data.get('uniParcId', "")
                
            if "oldestCrossRefCreated" in uniprot_data:
                data_format["info_created"] = uniprot_data.get("oldestCrossRefCreated", "")
            if "mostRecentCrossRefUpdated" in uniprot_data:
                data_format["info_modified"] = uniprot_data.get("mostRecentCrossRefUpdated", "")
                
            if "uniParcCrossReferences" in uniprot_data:         
                data_format["cross_references"] = uniprot_data.get("uniParcCrossReferences", [])
                for cross_ref in uniprot_data.get("uniParcCrossReferences", []):
                    if cross_ref.get("active", False) and "organism" in cross_ref:
                        if "geneName" in cross_ref:
                            gene_name = cross_ref.get("geneName", "")
                            data_format["associated_genes"] = gene_name
                        
                        organism = cross_ref.get("organism", {})
                        data_format["organism_scientific_name"] = organism.get("scientificName", "")
                        data_format["taxon_id"] = organism.get("taxonId", "")
                    
            if "sequence" in uniprot_data:
                sequence = uniprot_data.get("sequence", {})
                data_format["sequence_length"] = sequence.get("length", 0)
                data_format["sequence_mass"] = sequence.get("molWeight", 0)
                data_format["sequence_sequence"] = sequence.get("value", 0)
                
            if "features" in uniprot_data:
                features = uniprot_data.get("sequenceFeatures", [])
                data_format["features"] = features
                
                
        return data_format

    def fetch_and_process(self, df, pdb_code_column, function_file):
        
        
        function_file_path = function_file + ".csv"
        # Check if the old file exists
        from datetime import datetime
        if os.path.exists(function_file_path):
            # Rename the file
            current_date = datetime.now().strftime('%Y-%m-%d')
            new_data_function_file = function_file + "_" + current_date + ".csv"
            os.rename(function_file_path, new_data_function_file)
            print(f"File has been renamed from '{function_file_path}' to '{new_data_function_file}'.")
        else:
            print(f"File '{function_file_path}' does not exist.")
            
        # Load existing function data
        try:
            function_data = pd.read_csv(function_file_path)
        except FileNotFoundError:
            function_data = pd.DataFrame(columns=[
                "uniprot_id", "pdb_code", "uniProtkb_id", "info_type", "info_created", "info_modified", 
                "info_sequence_update", "annotation_score", "taxon_id", "organism_scientific_name", 
                "organism_common_name", "organism_lineage", "secondary_accession", "protein_recommended_name", 
                "protein_alternative_name", "associated_genes", "comment_function", "comment_interactions",
                "comment_catalytic_activity", "comment_subunit", "comment_PTM", "comment_caution", 
                "comment_subcellular_locations", "comment_alternative_products", 
                "comment_disease_name", "comment_disease", "comment_similarity", "features", "references", "keywords", 
                "sequence_length", "sequence_mass", "extra_attributes", "sequence_sequence", "molecular_function", 
                "cellular_component", "biological_process", "cross_references"
            ])
        
        # Initialize data dictionary
        data = {col: [] for col in function_data.columns}
        
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            pdb_code = row["Pdb Code"]
            uniprot_id = row[pdb_code_column]
            # Fetch data from UniProt for the given PDB code
            data_source, uniprot_data_list = self.fetch_data(uniprot_id)
            
            if data_source == "uniprotKB":
                for uniprot_data in uniprot_data_list:
                    extracted_data = self.extract_information(uniprot_data, data_source)
                    
                    # Append PDB code to extracted data
                    extracted_data['pdb_code'] = pdb_code
                    extracted_data['data_source'] = data_source
                    
                    for key in data:
                        data[key].append(extracted_data.get(key, None))
            else:
                extracted_data = self.extract_information(uniprot_data_list, data_source)
                    
                # Append PDB code to extracted data
                extracted_data['pdb_code'] = pdb_code
                extracted_data['data_source'] = data_source
                for key in data:
                    data[key].append(extracted_data.get(key, None))
                
                
        # Append data to function data DataFrame
        new_function_data = pd.DataFrame(data)
        function_data = pd.concat([function_data, new_function_data], ignore_index=True)
    
        function_data.to_csv(function_file_path, index=False)
        
        return function_data

# Load input data
data = pd.read_csv(modified_path + "/datasets/Quantitative_data.csv", low_memory=False)
data = data[data["uniprot_id"].notna()]
# data = data[data["Pdb Code"].isin(["1PTH", "8KH4", "2KIX"])]
# data = data[data["Pdb Code"].isin(["2KIX"])]
fetcher = UniProtDataFetcher()
result_df = fetcher.fetch_and_process(data, 'uniprot_id', modified_path + '/datasets/Uniprot_functions')

