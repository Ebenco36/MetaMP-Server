from abc import ABC, abstractmethod
import re
import importlib
import logging
import pandas as pd
from database.db import db
from sqlalchemy.inspection import inspect

logger = logging.getLogger(__name__)

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
        # Remove columns that start with 'Unnamed: 0.1'
        columns_to_exclude = [col for col in df.columns if col.startswith('Unnamed')]
        df = df.drop(columns=columns_to_exclude)
        # Identify columns to be removed
        columns_to_remove = [col for col in df.columns if '_id' in col.lower() and ('uniprot_id' not in col.lower())]
        
        # Remove identified columns
        df = df.drop(columns=columns_to_remove)
        
        return df

    @staticmethod
    def is_meaningful_value(value):
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        try:
            return not pd.isna(value)
        except TypeError:
            return True

    @staticmethod
    def get_record_key(model_class, row_dict):
        return tuple(
            str(row_dict.get(field_name) or "").strip()
            for field_name in Migration.get_key_fields(model_class)
        )

    @staticmethod
    def get_record_key_from_instance(model_class, record):
        return tuple(
            str(getattr(record, field_name, "") or "").strip()
            for field_name in Migration.get_key_fields(model_class)
        )

    @staticmethod
    def get_key_fields(model_class):
        if model_class.__name__ == "Uniprot":
            return ("uniprot_id", "pdb_code")
        if model_class.__name__ in {"MPSTURC", "MembraneProteinData"}:
            return ("pdb_code", "group", "subgroup")
        return ("pdb_code",)

    @staticmethod
    def get_lookup_filters(model_class, row_dict):
        return {
            field_name: row_dict.get(field_name)
            for field_name in Migration.get_key_fields(model_class)
        }

    @staticmethod
    def drop_duplicate_rows(df, model_class):
        key_fields = [
            field_name for field_name in Migration.get_key_fields(model_class)
            if field_name in df.columns
        ]
        if not key_fields:
            return df

        normalized = df.copy()
        for field_name in key_fields:
            normalized[field_name] = normalized[field_name].fillna("").astype(str).str.strip()

        keep_mask = ~(normalized.duplicated(subset=key_fields, keep="first"))
        return df.loc[keep_mask].reset_index(drop=True)

    @staticmethod
    def remove_duplicate_records(model_class, db):
        seen_keys = set()
        duplicate_records = []
        for record in model_class.query.order_by(model_class.id.asc()).all():
            record_key = Migration.get_record_key_from_instance(model_class, record)
            if all(not part for part in record_key):
                continue
            if record_key in seen_keys:
                duplicate_records.append(record)
                continue
            seen_keys.add(record_key)

        for record in duplicate_records:
            db.session.delete(record)

        if duplicate_records:
            logger.info(
                "Removed %s duplicate records from %s using natural key %s",
                len(duplicate_records),
                model_class.__tablename__,
                Migration.get_key_fields(model_class),
            )

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

        df = Migration.align_columns_to_model_keys(df, model_class)
        df = Migration.drop_duplicate_rows(df, model_class)
        return Migration.insert_or_update_records(df, model_class, db)

    @staticmethod
    def align_columns_to_model_keys(df, model_class):
        rename_map = {}
        mapper = inspect(model_class)
        for attribute in mapper.column_attrs:
            if not attribute.columns:
                continue
            attribute_key = attribute.key
            for column in attribute.columns:
                if (
                    column.name in df.columns
                    and attribute_key not in df.columns
                    and column.name != attribute_key
                ):
                    rename_map[column.name] = attribute_key
        if rename_map:
            df = df.rename(columns=rename_map)
        return df


    def insert_or_update_records(df, model_class, db):
        seen_keys = set()
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            record_key = Migration.get_record_key(model_class, row_dict)
            if all(not part for part in record_key):
                continue
            seen_keys.add(record_key)

            with db.session.no_autoflush:
                existing_record = (
                    model_class.query
                    .filter_by(**Migration.get_lookup_filters(model_class, row_dict))
                    .order_by(model_class.id.asc())
                    .first()
                )

            if existing_record:
                # If the record exists, update it
                for key, value in row.items():
                    if hasattr(existing_record, key):
                        current_value = getattr(existing_record, key)
                        if Migration.is_meaningful_value(value) or not Migration.is_meaningful_value(current_value):
                            setattr(existing_record, key, value)
            else:
                valid_keys = {attribute.key for attribute in inspect(model_class).column_attrs}
                filtered_dict = {k: v for k, v in row_dict.items() if k in valid_keys}
                new_record = model_class(**filtered_dict)
                # If the record doesn't exist, insert a new record
                # new_record = model_class(**row.to_dict())
                db.session.add(new_record)

        # Commit the changes
        logger.info("Committing %s dataset rows into %s", len(df), model_class.__tablename__)
        db.session.commit()
        Migration.remove_duplicate_records(model_class, db)
        db.session.commit()
        return seen_keys
    
    @staticmethod    
    def rename_id_column(df, new_name='new_id'):
        if 'id' in df.columns:
            df.rename(columns={'id': new_name}, inplace=True)
            print("Column 'id' renamed to '{}'".format(new_name))
        return df
