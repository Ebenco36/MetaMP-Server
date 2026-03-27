import re
from database.db import db

def get_column_type(dtype):
    # Map pandas datatypes to SQLAlchemy column types
    dtype_mapping = {
        'int64': db.Integer,
        'float64': db.Float,
        'object': db.Text,
        # Add more mappings as needed
    }
    return dtype_mapping.get(str(dtype), db.Text)

def shorten_column_name(column_name):
    # Split column name by '_' and select the first character of the first 4 words
    words = column_name.split('_')
    shortened_name = ''.join([word[:3] for word in words[:2]]) + '_'.join(words[2:]) if len(words) > 4 else column_name
    # Convert spaces to underscores
    cleaned_string = shortened_name.replace(' ', '_')
    
    # Remove special characters using regex
    cleaned_string = re.sub(r'[^a-zA-Z0-9_]', '', cleaned_string)
    return cleaned_string.lower()

