# import re
# import importlib
# import pandas as pd
# from datetime import datetime
# from database.db import db
# from src.Commands.Migration.classMigrate import Migration

# class MPSTRUCDatabase(Migration):

#     @staticmethod
#     def generate(csv_path, output_file='model.py'):
#         # Load CSV data to inspect headers and datatypes
#         df = pd.read_csv(csv_path, low_memory=False)
#         df = Migration.processColumns(df)
        
#         """
#         module_name = "src.MP.model_mpstruct"
#         class_name = "MPSTURC"
#         instance = create_instance(module_name, class_name)
#         if instance:
#             return instance
#         """
#         class MPSTURC(db.Model):
#             __table_args__ = {'extend_existing': True}
#             __tablename__ = 'membrane_protein_mpstruct'
#             id = db.Column(db.Integer, primary_key=True)

#         # Add columns dynamically based on CSV headers and datatypes
#         for column_name, dtype in zip(df.columns, df.dtypes):
            
#             max_length = None
#             if dtype == 'object':
#                 # Calculate the maximum length for string columns
#                 max_length = df[column_name].astype(str).apply(len).max()
#             column_type = Migration.get_column_type(dtype, max_length)
#             # Shorten the column name for compatibility
#             short_column_name = Migration.shorten_column_name(column_name)
#             if not hasattr(MPSTURC, short_column_name):
#                 setattr(MPSTURC, short_column_name, db.Column(column_type))
                
#         # Create the output file with the generated model class
#         with open(output_file, 'w') as file:
#             file.write("from datetime import datetime\n")
#             file.write("from database.db import db\n\n")
#             file.write(f"class {MPSTURC.__name__}(db.Model):\n")
#             file.write("    __tablename__ = 'membrane_protein_mpstruct'\n")
#             file.write("    id = db.Column(db.Integer, primary_key=True)\n")

#             # Add columns to the file
#             for column_name, dtype in zip(df.columns, df.dtypes):
#                 column_type = Migration.get_column_type(dtype)
#                 shortened_name = Migration.shorten_column_name(column_name)
#                 file.write(f"    {shortened_name} = db.Column(db.{column_type.__name__})\n")
                
#         print(f"Model class has been generated and saved to {output_file}")
#         return MPSTURC


# def generate_model_class_MPSTRUCT(csv_path, output_file='model.py'):
#     return MPSTRUCDatabase().generate(csv_path, output_file)



import pandas as pd
from datetime import datetime
from database.db import db
from src.Commands.Migration.classMigrate import Migration

class MPSTRUCDatabase(Migration):

    @staticmethod
    def generate(csv_path, output_file='model.py'):
        df = pd.read_csv(csv_path, low_memory=False)
        df = Migration.processColumns(df)

        max_lengths = {}

        class MPSTURC(db.Model):
            __table_args__ = {'extend_existing': True}
            __tablename__ = 'membrane_protein_mpstruct'
            id = db.Column(db.Integer, primary_key=True)

        for column_name, dtype in zip(df.columns, df.dtypes):
            short_column_name = Migration.shorten_column_name(column_name)
            if hasattr(MPSTURC, short_column_name):
                continue

            if dtype == 'object':
                raw_max = df[column_name].astype(str).apply(len).max()
                max_lengths[short_column_name] = raw_max
                if raw_max is None or raw_max > 255:
                    column_type = db.Text
                else:
                    column_type = db.String(int(raw_max))
            else:
                column_type = Migration.get_column_type(dtype)

            setattr(MPSTURC, short_column_name, db.Column(column_type))

        with open(output_file, 'w') as file:
            file.write("from datetime import datetime\n")
            file.write("from database.db import db\n\n")
            file.write(f"class {MPSTURC.__name__}(db.Model):\n")
            file.write("    __tablename__ = 'membrane_protein_mpstruct'\n")
            file.write("    id = db.Column(db.Integer, primary_key=True)\n")

            for column_name, dtype in zip(df.columns, df.dtypes):
                short_name = Migration.shorten_column_name(column_name)

                if dtype == 'object':
                    max_len = max_lengths.get(short_name, None)
                    if max_len is None or max_len > 55:
                        col_type_str = "Text"
                    else:
                        col_type_str = f"String({int(max_len)})"
                else:
                    col_type_str = Migration.get_column_type(dtype).__name__

                file.write(f"    {short_name} = db.Column(db.{col_type_str})\n")

        print(f"Model class has been generated and saved to {output_file}")
        return MPSTURC


def generate_model_class_MPSTRUCT(csv_path, output_file='model.py'):
    return MPSTRUCDatabase().generate(csv_path, output_file)
