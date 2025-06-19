import os
import requests
import pandas as pd
import datetime
from math import ceil

# Base directories
DATASETS_PATH = os.path.join('.', 'datasets')
BATCH_PATH = os.path.join(DATASETS_PATH, 'UniProt')

class UniProtDataFetcher:
    def __init__(self):
        # Endpoints
        self.search_url       = "https://rest.uniprot.org/uniprotkb/search"
        self.uniparc_url      = "https://rest.uniprot.org/uniparc/search"
        # HTTP headers
        self.headers = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "Mozilla/5.0"
        }

    def create_directories(self):
        os.makedirs(DATASETS_PATH, exist_ok=True)
        os.makedirs(BATCH_PATH, exist_ok=True)

    def fetch_and_save(self, df: pd.DataFrame, id_col: str):
        """
        Batch-fetch full UniProt data for IDs in df[id_col], in groups of 200.
        Saves per-batch CSVs to datasets/UniProt/, then merges into one dated file.
        """
        self.create_directories()
        today       = datetime.date.today().strftime('%Y-%m-%d')
        master_name = f"Uniprot_functions_{today}.csv"
        master_fp   = os.path.join(DATASETS_PATH, master_name)

        ids         = df[id_col].dropna().astype(str).tolist()
        total       = len(ids)
        batch_count = ceil(total / 200)

        print(f"Total IDs: {total}; batching into {batch_count} of 200 each.")

        # 1) process each batch
        for batch_idx in range(batch_count):
            start = batch_idx * 200
            end   = min(start + 200, total)
            batch_ids = ids[start:end]

            batch_name = f"Uniprot_functions_{today}_batch{batch_idx+1}.csv"
            batch_fp   = os.path.join(BATCH_PATH, batch_name)
            if os.path.exists(batch_fp):
                print(f"✓ Batch {batch_idx+1} exists; skipping.")
                continue

            print(f"→ Fetching batch {batch_idx+1}: IDs {start+1}-{end}")
            records = []

            for uid in batch_ids:
                # primary search
                url = f"{self.search_url}?query={uid}&format=json"
                r = requests.get(url, headers=self.headers)
                if r.status_code != 200:
                    print(f"  ✗ Search failed for {uid}")
                    continue
                results = r.json().get('results', [])
                if results:
                    for entry in results:
                        rec = self.extract_information(entry, source='uniprotKB')
                        rec['uniprot_id'] = uid
                        records.append(rec)
                else:
                    # fallback to UniParc
                    up = self.fetch_from_uniparc(uid)
                    if up:
                        rec = self.extract_information(up, source='UniParc')
                        rec['uniprot_id'] = uid
                        records.append(rec)
                    else:
                        print(f"  ⚠ No data for {uid}")

            if records:
                pd.DataFrame(records).to_csv(batch_fp, index=False)
                print(f"★ Saved {batch_name}")
            else:
                print(f"⚠ Empty batch {batch_idx+1}")

        # 2) merge batches
        batch_files = sorted(
            f for f in os.listdir(BATCH_PATH)
            if f.startswith(f"Uniprot_functions_{today}_batch") and f.endswith('.csv')
        )
        if not batch_files:
            print("No batch files to merge.")
            return None

        dfs = []
        for fn in batch_files:
            dfs.append(pd.read_csv(os.path.join(BATCH_PATH, fn)))
        master_df = pd.concat(dfs, ignore_index=True)
        master_df.to_csv(master_fp, index=False)
        print(f"Combined into {master_name}")
        return master_df

    def fetch_from_uniparc(self, uid: str):
        url = f"{self.uniparc_url}?query={uid}&format=json"
        r = requests.get(url, headers=self.headers)
        if r.status_code == 200:
            res = r.json().get('results', [])
            return res[0] if res else None
        return None

    def extract_information(self, data: dict, source: str = 'uniprotKB') -> dict:
        # Template with all fields
        rec = {
            'data_source': source,
            'uniprot_id': '',
            'uniProtkb_id': '',
            'info_type': '',
            'info_created': '',
            'info_modified': '',
            'info_sequence_update': '',
            'annotation_score': '',
            'taxon_id': '',
            'organism_scientific_name': '',
            'organism_common_name': '',
            'organism_lineage': '',
            'secondary_accession': '',
            'protein_recommended_name': '',
            'protein_alternative_name': '',
            'associated_genes': '',
            'comment_function': [],
            'comment_interactions': [],
            'comment_catalytic_activity': [],
            'comment_subunit': [],
            'comment_PTM': [],
            'comment_caution': [],
            'comment_tissue_specificity': [],
            'comment_subcellular_locations': [],
            'comment_alternative_products': [],
            'comment_disease_name': '',
            'comment_disease': [],
            'comment_similarity': [],
            'features': [],
            'references': [],
            'cross_references': [],
            'keywords': [],
            'sequence_length': 0,
            'sequence_mass': 0,
            'sequence_sequence': '',
            'extra_attributes': [],
            'molecular_function': '',
            'cellular_component': '',
            'biological_process': ''
        }
        if source == 'uniprotKB':
            rec['uniprot_id'] = data.get('primaryAccession','')
            rec['uniProtkb_id'] = data.get('uniProtkbId','')
            rec['info_type'] = data.get('entryType','')
            audit = data.get('entryAudit',{})
            rec['info_created'] = audit.get('firstPublicDate','')
            rec['info_modified'] = audit.get('lastAnnotationUpdateDate','')
            rec['info_sequence_update'] = audit.get('lastSequenceUpdateDate','')
            rec['annotation_score'] = data.get('annotationScore','')
            org = data.get('organism',{})
            rec['taxon_id'] = org.get('taxonId','')
            rec['organism_scientific_name'] = org.get('scientificName','')
            rec['organism_common_name'] = org.get('commonName','')
            rec['organism_lineage'] = ';'.join(org.get('lineage',[]))
            rec['secondary_accession'] = ';'.join(data.get('secondaryAccessions',[]))
            pd_desc = data.get('proteinDescription',{})
            rec['protein_recommended_name'] = pd_desc.get('recommendedName',{}).get('fullName',{}).get('value','')
            rec['protein_alternative_name'] = ';'.join(
                [a.get('fullName',{}).get('value','') for a in pd_desc.get('alternativeNames',[])]
            )
            rec['associated_genes'] = ';'.join(
                [g.get('geneName',{}).get('value','') for g in data.get('genes',[])]
            )
            for c in data.get('comments',[]):
                t = c.get('commentType','')
                if t=='FUNCTION': rec['comment_function'] = c.get('texts',[])
                if t=='INTERACTION': rec['comment_interactions'] = c.get('interactions',[])
                if t=='SUBCELLULAR LOCATION': rec['comment_subcellular_locations'] = c.get('subcellularLocations',[])
                if t=='ALTERNATIVE PRODUCTS': rec['comment_alternative_products'] = c.get('isoforms',[])
                if t=='DISEASE':
                    rec['comment_disease'].append(c.get('disease',{}))
                if t=='PTM': rec['comment_PTM'] = c.get('texts',[])
                if t=='CATALYTIC ACTIVITY': rec['comment_catalytic_activity'].append(c.get('reaction',{}))
            rec['comment_disease_name'] = ';'.join([d.get('diseaseId','') for d in rec['comment_disease']])
            rec['features'] = data.get('features',[])
            rec['references'] = data.get('references',[])
            rec['keywords'] = data.get('keywords',[])
            seq = data.get('sequence',{})
            rec['sequence_length'] = seq.get('length',0)
            rec['sequence_mass'] = seq.get('molWeight',0)
            rec['sequence_sequence'] = seq.get('value','')
            rec['extra_attributes'] = data.get('uniProtKBCrossReferences',[])
            # cross refs and GO terms
            go = {'F':'molecular_function','C':'cellular_component','P':'biological_process'}
            for x in data.get('uniProtKBCrossReferences',[]):
                for p in x.get('properties',[]):
                    v = p.get('value','')
                    if ':' in v and v[1]==':':
                        key,term = v.split(':',1)
                        fld = go.get(key)
                        if fld: rec[fld]+= term+';'
            for fld in ['molecular_function','cellular_component','biological_process']:
                rec[fld] = rec[fld].strip(';')
        else:
            # UniParc fallback
            rec['uniprot_id'] = data.get('uniParcId','')
            rec['info_created'] = data.get('oldestCrossRefCreated','')
            rec['info_modified'] = data.get('mostRecentCrossRefUpdated','')
            rec['cross_references'] = data.get('uniParcCrossReferences',[])
        return rec

# Example usage
if __name__=='__main__':
    df = pd.read_csv(os.path.join(DATASETS_PATH,'Quantitative_data.csv'), low_memory=False)
    fetcher = UniProtDataFetcher()
    combined_df = fetcher.fetch_and_save(df, 'uniprot_id')
    # print(combined_df.head())
