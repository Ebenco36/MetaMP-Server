from abc import ABC, abstractmethod
import re
import importlib
import pandas as pd
from database.db import db
from sqlalchemy.inspection import inspect

# Define an abstract class
class Migration(ABC):
    
    @abstractmethod
    def generate(self):
        pass
    
    @staticmethod
    def create_instance(module_name, class_name):
        try:
            # Import the module dynamically
            module = importlib.import_module(module_name)

            # Get the class from the module
            if hasattr(module, class_name):
                class_ = getattr(module, class_name)
                # Create an instance of the class with optional arguments
                instance = class_
                return instance
            else:
                return False
        except ImportError:
            return False


    def get_column_type(dtype, max_length=None):
        """
        Map pandas data types to SQLAlchemy data types.

        Parameters:
        dtype (str): The pandas data type as a string.
        max_length (int, optional): Maximum length of string columns to decide between String and Text.

        Returns:
        SQLAlchemy type: Corresponding SQLAlchemy data type.
        """
        # Map pandas data types to SQLAlchemy data types
        type_mapping = {
            'int64': db.Integer,
            'float64': db.Float,
            'datetime64[ns]': db.DateTime,
            'bool': db.Boolean,
            'timedelta64[ns]': db.Interval,
        }
        
        dtype_str = str(dtype)
        if dtype_str == 'object':
            if max_length is not None and max_length <= 255:
                return db.String(max_length)  # Use String with a max length if specified
            else:
                return db.Text  # Use Text for larger or undefined length
        
        return type_mapping.get(dtype_str, db.String)  # Default to String if dtype is not found

    def old_get_column_type(dtype):
        # Map pandas data types to SQLAlchemy data types
        type_mapping = {
            'int64': db.Integer,
            'float64': db.Float,
            'datetime64[ns]': db.DateTime,
            'object': db.String
        }
        return type_mapping.get(str(dtype), db.String)

    def shorten_column_name(column_name, strip_special_character=True):
        # Split column name by '_' and select the first character of the first 4 words
        words = column_name.split('_')
        shortened_name = ''.join([word[:3] for word in words[:2]]) + '_'.join(words[2:]) if len(words) > 4 else column_name
        # Convert spaces to underscores
        cleaned_string = shortened_name.replace(' ', '_')
        
        if(strip_special_character):
            # Remove special characters using regex
            cleaned_string = re.sub(r'[^a-zA-Z0-9_]', '', cleaned_string)
        return cleaned_string.lower()


    def processColumns(df):
        print("here is the length: " + str(len(df)))
        # Remove columns that start with 'Unnamed: 0.1'
        columns_to_exclude = [col for col in df.columns if col.startswith('Unnamed')]
        df = df.drop(columns=columns_to_exclude)
        # Identify columns to be removed
        columns_to_remove = [col for col in df.columns if '_id' in col.lower() and ('uniprot_id' not in col.lower())]
        
        # Remove identified columns
        df = df.drop(columns=columns_to_remove)
        
        return df
    
    def load_csv_data(model_class, csv_path):
        # Load CSV data into a pandas DataFrame
        df = pd.read_csv(csv_path, low_memory=False)
        
        if (model_class.__name__ == "Uniprot"):
            df = df[df['uniprot_id'].notna()]
        if (model_class.__name__ == "OPM"):
            df = Migration.rename_id_column(df, "opm_id")
        
        df = Migration.processColumns(df)
        # Loop through columns and update names.
        # We are doing this because of the issue with the column length
        for old_name in df.columns:
            # Create a new name (you can modify this logic based on your requirements)
            new_name = Migration.shorten_column_name(old_name)
            # Rename the column
            df = df.rename(columns={old_name: new_name})

        Migration.insert_or_update_records(df, model_class, db)


    def insert_or_update_records(df, model_class, db):
        for index, row in df.iterrows():
            if (model_class.__name__ == "Uniprot"):
                #  or "uniprot_id" in row.index
                uniprot_id = row['uniprot_id']
                pdb_code = row['pdb_code']
                # Check if the record with the given PDB code already exists
                existing_record = model_class.query.filter_by(uniprot_id=uniprot_id, pdb_code=pdb_code).first()
            else:
                pdb_code = row['pdb_code']
                # Check if the record with the given PDB code already exists
                existing_record = model_class.query.filter_by(pdb_code=pdb_code).first()

            if existing_record:
                # If the record exists, update it
                for key, value in row.items():
                    if hasattr(existing_record, key):
                        setattr(existing_record, key, value)
            else:
                valid_keys = {c.key for c in inspect(model_class).c}
                filtered_dict = {k: v for k, v in row.to_dict().items() if k in valid_keys}
                new_record = model_class(**filtered_dict)
                # If the record doesn't exist, insert a new record
                # new_record = model_class(**row.to_dict())
                db.session.add(new_record)

        # Commit the changes
        db.session.commit()
    
    @staticmethod    
    def rename_id_column(df, new_name='new_id'):
        if 'id' in df.columns:
            df.rename(columns={'id': new_name}, inplace=True)
            print("Column 'id' renamed to '{}'".format(new_name))
        return df