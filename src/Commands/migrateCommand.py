import re
import pandas as pd
import importlib
import logging
from datetime import datetime
from database.db import db
from src.Commands.Migration.classMigrate import Migration

class enrichedMPStrucDB(Migration):
    @staticmethod
    def generate(csv_path, output_file='model.py'):
        # Load CSV data to inspect headers and datatypes
        df = pd.read_csv(csv_path, low_memory=False)
        df = Migration.processColumns(df)
        
        """NOT NEEDED
        module_name = "src.MP.model"
        class_name = "MembraneProteinData"
        instance = create_instance(module_name, class_name)
        if instance:
            return instance
        """
        
        class MembraneProteinData(db.Model):
            __table_args__ = {'extend_existing': True}
            __tablename__ = 'membrane_proteins'
            id = db.Column(db.Integer, primary_key=True)

        # Add columns dynamically based on CSV headers and datatypes
        
        for column_name, dtype in zip(df.columns, df.dtypes):
            
            max_length = None
            if dtype == 'object':
                # Calculate the maximum length for string columns
                max_length = df[column_name].astype(str).apply(len).max()
            column_type = Migration.get_column_type(dtype, max_length)
            # Shorten the column name for compatibility
            short_column_name = Migration.shorten_column_name(column_name)
            if not hasattr(MembraneProteinData, short_column_name):
                setattr(MembraneProteinData, short_column_name, db.Column(column_type))
                
        # for column_name, dtype in zip(df.columns, df.dtypes):
        #     column_type = Migration.get_column_type(dtype)
        #     if not hasattr(MembraneProteinData, Migration.shorten_column_name(column_name)):
        #         setattr(MembraneProteinData, Migration.shorten_column_name(column_name), db.Column(column_type))
                
                
        # Create the output file with the generated model class
        with open(output_file, 'w') as file:
            file.write("from datetime import datetime\n")
            file.write("from database.db import db\n\n")
            file.write(f"class {MembraneProteinData.__name__}(db.Model):\n")
            file.write("    __tablename__ = 'membrane_proteins'\n")
            file.write("    id = db.Column(db.Integer, primary_key=True)\n")

            # Add columns to the file
            for column_name, dtype in zip(df.columns, df.dtypes):
                column_type = Migration.get_column_type(dtype)
                shortened_name = Migration.shorten_column_name(column_name)
                if not hasattr(MembraneProteinData, shortened_name):
                    print("still here.........")
                    file.write(f"    {shortened_name} = db.Column(db.{column_type.__name__})\n")
                else:
                    print(f"Attribute {shortened_name} already exists on MembraneProteinData.")
                    protein = MembraneProteinData()
                    setattr(protein, shortened_name, None)
                    delattr(protein, shortened_name)
                    file.write(f"    {shortened_name} = db.Column(db.{column_type.__name__})\n")
            
            file.write("    TMbed_tm_count = db.Column(db.Integer)\n")
            file.write("    DeepTMHMM_tm_count = db.Column(db.Integer)\n")
            file.write("    created_at = db.Column(db.DateTime, default=datetime.utcnow)\n")
            file.write("    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)\n")

        print(f"Model class has been generated and saved to {output_file}")
        return MembraneProteinData
    

def generate_model_class(csv_path, output_file='model.py'):
    return enrichedMPStrucDB().generate(csv_path, output_file)