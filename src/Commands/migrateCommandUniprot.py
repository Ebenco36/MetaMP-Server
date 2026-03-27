import re
import importlib
import pandas as pd
import numpy as np
from datetime import datetime
from database.db import db
from sqlalchemy.exc import ProgrammingError
from src.Commands.Migration.classMigrate import Migration


class UniprotDatabase(Migration):
    @staticmethod
    def generate(csv_path, output_file='model.py'):
        try:
            # Load CSV data to inspect headers and datatypes
            df = pd.read_csv(csv_path, low_memory=False)
            # Remove rows where 'uniprot_id' is NaN and replace NaN with None' is None
            df = df[df['uniprot_id'].notna()]
            
            df = Migration.processColumns(df)
            #df = Migration.rename_id_column(df, "uniprot_id")

            """
            module_name = "src.MP.model_uniprot"
            class_name = "Uniprot"
            instance = create_instance(module_name, class_name)
            if instance:
                return instance
            """
            
            class Uniprot(db.Model):
                __table_args__ = {'extend_existing': True}
                __tablename__ = 'membrane_protein_uniprot'
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
                if not hasattr(Uniprot, short_column_name):
                    setattr(Uniprot, short_column_name, db.Column(column_type))
                
            # Create the output file with the generated model class
            with open(output_file, 'w') as file:
                file.write("from datetime import datetime\n")
                file.write("from database.db import db\n\n")
                file.write(f"class {Uniprot.__name__}(db.Model):\n")
                file.write("    __tablename__ = 'membrane_protein_uniprot'\n")
                file.write("    id = db.Column(db.Integer, primary_key=True)\n")

                # Add columns to the file
                for column_name, dtype in zip(df.columns, df.dtypes):
                    column_type = Migration.get_column_type(dtype)
                    shortened_name = Migration.shorten_column_name(column_name)
                    file.write(f"    {shortened_name} = db.Column(db.{column_type.__name__})\n")
                    
            print(f"Model class has been generated and saved to {output_file}")
            return Uniprot
        except ProgrammingError as e:
            print(f"An error occurred while generating the model class: {e}")


def generate_model_class_Uniprot(csv_path, output_file='model.py'):
    return UniprotDatabase().generate(csv_path, output_file)