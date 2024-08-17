import os
import pandas as pd
import urllib
import shutil
import datetime
import xml.etree.ElementTree as et

data_path = os.environ.get("AIRFLOW_HOME")
if (data_path):
    modified_path = data_path.replace("/airflow_home", "")
else:
    modified_path = "."

class MPSTRUCT:

    def create_directory(self, directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_path}' already exists.")
        
    def load_data(self):
        # create directory
        self.create_directory("datasets")
        #Fetch update timne of the database from the thml content of the mpstruc website
        req = urllib.request.urlopen("https://blanco.biomol.uci.edu/mpstruc/")
        page = req.read()
        page = str(page)
        update_time_marker = page.find("Last database update:")
        update_time = page[update_time_marker+22:update_time_marker+21+25]


        #Get the current time
        current_time = datetime.datetime.now()
        year = current_time.year
        day = current_time.day
        month = current_time.month
        hour = current_time.hour

        #Split the update time string and get the specific times out of it
        upd_times = update_time.split()

        upd_year = int(upd_times[2])
        upd_day = int(upd_times[0]) 
        upd_month = upd_times[1]
        upd_hour =int(upd_times[4][0:2])

            
        upd_month = self.convert_month(upd_month)

        #Introduce the update token for flexibility, so that it can easily be set to True if the database is to be updated without meeint the conditions
        update_token = True

        if ((year >= upd_year) & (day >= upd_day) & (month >= upd_month) & (hour >= upd_hour)):
            update_token = True


        #Write the updated mpstruc into a new file
        new_mpstruc = urllib.request.urlopen("https://blanco.biomol.uci.edu/mpstruc/listAll/mpstrucTblXml")    

        current_date = datetime.date.today().strftime('%Y-%m-%d')
        if update_token:
            with open(modified_path + "/datasets/mpstrucTblXml.xml", "wb") as outfile:
                shutil.copyfileobj(new_mpstruc, outfile)
        print("What are we doing here: " + modified_path + "/datasets/mpstrucTblXml.xml")
        return self


    def parse_data(self):
        # create directory
        self.create_directory("datasets")
        #Parse mpstruc xml received from 'Mpstuc Update' as an element tree
        current_date = datetime.date.today().strftime('%Y-%m-%d')
        tree = et.parse(modified_path + "/datasets/mpstrucTblXml.xml")
        root = tree.getroot()

        #Start the long journey of generating the .csv-table
        protein_entries = []

        for groups in root:
            for group in groups:
                group_name = group[0].text
                for subgroup in group[2]:
                    subgroup_name = subgroup[0].text
                    for protein in subgroup[1]:
                        pdbCode = protein[0].text
                        name = protein[1].text
                        species = protein[2].text
                        taxonomicDomain = protein[3].text
                        expressedInSpecies = protein[4].text
                        resolution = protein[5].text
                        description = protein[6].text
                        bibliography = []
                        for i in range(len(protein[7])):
                            bibliography.append([protein[7][i].tag,protein[7][i].text])
                        secondaryBibliographies = protein[8].text
                        relatedPdbEntries = protein[9].text
                        memberProteins = []
                        m_protein_entries = []
                        for memberprotein in protein[10]:
                            memberProteins.append([memberprotein[0].tag, memberprotein[0].text])
                            
                            m_pdbCode = memberprotein[0].text
                            m_masterProteinPdbCode = memberprotein[1].text
                            m_name = memberprotein[2].text
                            m_species = memberprotein[3].text
                            m_taxonomicDomain = memberprotein[4].text
                            m_expressedInSpecies = memberprotein[5].text
                            m_resolution = memberprotein[6].text
                            m_description = memberprotein[7].text
                            m_bibliography = []
                            for j in range(len(memberprotein[8])):
                                m_bibliography.append([memberprotein[8][j].tag, memberprotein[8][j].text])
                            m_secondaryBibliographies = memberprotein[9].text
                            m_relatedPdbEntries = memberprotein[10].text
                            m_protein_entry = [group_name,subgroup_name, m_pdbCode, m_masterProteinPdbCode, m_name, m_species, m_taxonomicDomain, m_expressedInSpecies, m_resolution, m_description, m_bibliography, m_secondaryBibliographies, m_relatedPdbEntries] 
                            m_protein_entries.append(m_protein_entry)
                            
                        protein_entry = [group_name,subgroup_name, pdbCode, "MasterProtein", name, species, taxonomicDomain, expressedInSpecies, resolution, description, bibliography, secondaryBibliographies, relatedPdbEntries, memberProteins]
                        protein_entries.append(protein_entry)
                        for one_entry in m_protein_entries:
                            protein_entries.append(one_entry)
                        m_protein_entries = []
                        
        data = pd.DataFrame(protein_entries, columns = ["Group","Subgroup","Pdb Code","Is Master Protein?","Name","Species","Taxonomic Domain","Expressed in Species","Resolution","Description","Bibliography","Secondary Bibliogrpahies","Related Pdb Entries","Member Proteins"])
        current_date = datetime.date.today().strftime('%Y-%m-%d')
        data.to_csv(modified_path + "/datasets/Mpstruct_dataset.csv", index=False)

        #Save the unique Codes to know which proteins to fetch from the PDB
        mpstruck_ids = data["Pdb Code"]
        mpstruck_ids.to_csv(modified_path + "/datasets/mpstruct_ids.csv", index=False)
        
        
    def fetch_data(self):
        return self.load_data().parse_data()
    
    
    def convert_month(self, mon):
        if (mon == "Jan"):
            return 1
        if (mon == "Feb"):
            return 2
        if (mon == "Mar"):
            return 3
        if (mon == "Apr"):
            return 4
        if (mon == "May"):
            return 5
        if (mon == "Jun"):
            return 6
        if (mon == "Jul"):
            return 7
        if (mon == "Aug"):
            return 8
        if (mon == "Sep"):
            return 9
        if (mon == "Oct"):
            return 10
        if (mon == "Nov"):
            return 11
        if (mon == "Dec"):
            return 12
    


# Instantiate the class and call the function
mpstruct_obj = MPSTRUCT()
mpstruct_obj.fetch_data()