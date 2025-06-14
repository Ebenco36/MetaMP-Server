import random
import numpy as np
import pandas as pd
import altair as alt
from flask import abort
from database.db import db
from sqlalchemy import func
from datetime import timedelta
from sqlalchemy.sql import select
from sqlalchemy import or_, and_
from src.MP.model_opm import OPM
from collections import OrderedDict
from src.services.graphs.helpers import convert_chart
from utils.redisCache import RedisCache
from src.MP.model_uniprot import Uniprot
from sqlalchemy.orm import Query, aliased
from sqlalchemy import select, func, Table
from sqlalchemy.exc import SQLAlchemyError
from src.MP.model import MembraneProteinData
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

def get_all_items():
    return MembraneProteinData.query

def apply_search_and_filter(query, search_terms, MP, OP, UP):
    data = search_terms.get("search_terms", {})
    search_term = data.get('search_term', None)
    group = data.get('group', None)
    subgroup = data.get('subgroup', None)
    taxonomic_domain = data.get('taxonomic_domain', None)
    experimental_method = data.get('experimental_methods', None)
    molecular_function = data.get('molecular_function', None)
    cellular_component = data.get('cellular_component', None)
    biological_process = data.get('biological_process', None)
    family_name = data.get('family_name', None)
    species = data.get('species', None)
    membrane_name = data.get('membrane_name', None)
    super_family = data.get('super_family', None)
    superfamily_classtype_name = data.get('super_family', None)
    
    conditions = []

    # Check and add conditions for MembraneProteinData
    if search_term:
        conditions.append(
            or_(
                MP.name.ilike(f"%{search_term}%"),
                MP.pdb_code.ilike(f"%{search_term}%"),
                UP.uniprot_id.ilike(f"%{search_term}%"),
                UP.comment_disease.ilike(f"%{search_term}%"),
            )
        )
    if group and group != "All":
        conditions.append(MP.group.ilike(f"%{group}%"))

    if subgroup and subgroup != "All":
        conditions.append(MP.subgroup.ilike(f"%{subgroup}%"))

    if taxonomic_domain and taxonomic_domain != "All":
        conditions.append(MP.taxonomic_domain.ilike(f"%{taxonomic_domain}%"))

    if experimental_method and experimental_method != "All":
        conditions.append(MP.rcsentinfo_experimental_method.ilike(f"%{experimental_method}%"))

    
    # Check and add conditions for OPM
    if family_name and family_name != "All":
        conditions.append(OP.family_name_cache.ilike(f"%{family_name}%"))

    if species and species != "All":
        conditions.append(OP.species_name_cache.ilike(f"%{species}%"))

    if membrane_name and membrane_name != "All":
        conditions.append(OP.membrane_name_cache.ilike(f"%{membrane_name}%"))

    if super_family and super_family != "All":
        conditions.append(OP.family_superfamily_name.ilike(f"%{super_family}%"))

    if superfamily_classtype_name and superfamily_classtype_name != "All":
        conditions.append(OP.family_superfamily_classtype_name.ilike(f"%{superfamily_classtype_name}%"))

    # Check and add conditions for Uniprot
    if molecular_function and molecular_function != "All":
        conditions.append(UP.molecular_function.ilike(f"%{molecular_function}%"))

    if cellular_component and cellular_component != "All":
        conditions.append(UP.cellular_component.ilike(f"%{cellular_component}%"))

    if biological_process and biological_process != "All":
        conditions.append(UP.biological_process.ilike(f"%{biological_process}%"))

    # Compose final query using all non-empty conditions
    if conditions:
        query = query.filter(and_(*conditions))
    
    # Print out the final SQL query
    # print(str(query.statement.compile(db.engine, compile_kwargs={"literal_binds": True})))
    
    return query

def apply_sorting(query, sort_by, sort_order, MP, OP, UP):
    if sort_by:
        # Determine which alias the sort column belongs to
        if hasattr(MP, sort_by):
            column = getattr(MP, sort_by)
        elif hasattr(OP, sort_by):
            column = getattr(OP, sort_by)
        elif hasattr(UP, sort_by):
            column = getattr(UP, sort_by)
        else:
            raise ValueError(f"Unknown sort_by column: {sort_by}")
        
        query = query.order_by(column.asc() if sort_order.lower() == 'asc' else column.desc())
    # print(query.all())
    return query

def get_columns_excluding(model, exclude_columns):
    return [column for column in model.__table__.columns if column.name not in exclude_columns]


def get_items(request = {}):
    data = request.get("search_terms", {})
    page = int(request.get("page", 10))
    per_page = int(request.get('per_page', 10))
    sort_by = request.get('sort_by', "id")
    sort_order = request.get('sort_order', "asc")
    download = request.get('download', "")
    # Define aliases
    MP = aliased(MembraneProteinData, name='mp')
    OP = aliased(OPM, name='op')
    UP = aliased(Uniprot, name='up')
    
    query = db.session.query(MP, OP, UP)\
        .outerjoin(OP, OP.pdb_code == MP.pdb_code)\
        .outerjoin(UP, UP.pdb_code == MP.pdb_code)


    # Apply key search item to filter
    query = apply_search_and_filter(query, request, MP, OP, UP)
    items = apply_sorting(query, sort_by, sort_order, MP, OP, UP)
    
    if download in ["csv", "xlsx"]:
        items = items.all()  # Get filtered and sorted records
        if items:
            def get_column_names(model):
                return [column.name for column in model.__table__.columns]
            
            membrane_protein_columns = get_column_names(MembraneProteinData)
            opm_columns = get_column_names(OPM)
            uniprot_columns = get_column_names(Uniprot)
            
            all_columns = ([(f"membrane_{col}", col) for col in membrane_protein_columns] +
                        [(f"opm_{col}", col) for col in opm_columns] +
                        [(f"uniprot_{col}", col) for col in uniprot_columns])
            
            records = []
            for item in items:
                mp_data, op_data, up_data = item
                record = {}
                for prefixed_col, col in all_columns:
                    if col in membrane_protein_columns:
                        record[prefixed_col] = getattr(mp_data, col, None)
                    elif col in opm_columns:
                        record[prefixed_col] = getattr(op_data, col, None) if op_data else None
                    elif col in uniprot_columns:
                        record[prefixed_col] = getattr(up_data, col, None) if up_data else None
                records.append(record)
            
            records_df = pd.DataFrame(records)
        else:
            records_df = pd.DataFrame()
            
        # records_df.to_csv("all_data.csv")
        return records_df

    # Pagination
    paginated_items = items.paginate(page=page, per_page=per_page, error_out=False)
    return extract_items_and_metadata(paginated_items, items.count(), MP, OP, UP)
    
    
def extract_items_and_metadata(paginated_items, total_item_count = 0, MP=None, OP=None, UP=None):
    # Compute total_columns from the result tuple
    remove_list = [
        "created_at", "updated_at", 
        "id"
    ]
    mp_columns = [column for column in MP.__table__.columns if column.name not in remove_list]
    op_columns = [column for column in OP.__table__.columns if column.name not in remove_list]
    up_columns = [column for column in UP.__table__.columns if column.name not in remove_list]
    
    # Combine all columns to save
    all_columns = {
        "MP": mp_columns,
        "OP": op_columns,
        "UP": up_columns
    }

    # Save column lists to a file
    file_path = 'filtered_columns.txt'

    with open(file_path, 'w') as file:
        for table_name, columns in all_columns.items():
            file.write(f"{table_name} columns:\n")
            for column in columns:
                file.write(f"  - {column}\n")
            file.write("\n")
        
    total_columns = len(mp_columns) + len(op_columns) + len(up_columns)
    
    # Extract items and metadata
    items_list = [OrderedDict([
        ('id', mp_data.id), 
        ('name', mp_data.name),
        ('group', mp_data.group),
        ('species', mp_data.species),
        ('subgroup', mp_data.subgroup),
        ('pdb_code', mp_data.pdb_code),
        ('uniprot_id', up_data.uniprot_id if up_data else None),
        ('resolution', mp_data.resolution),
        ('exptl_method', mp_data.exptl_method),
        ('taxonomic_domain', mp_data.taxonomic_domain),
        ('expressed_in_species', mp_data.expressed_in_species),
        ('rcsentinfo_experimental_method', mp_data.rcsentinfo_experimental_method),
        ('comment_disease_name', up_data.comment_disease_name if up_data else None),
    ]) for mp_data, op_data, up_data in paginated_items.items]
    result = {
        'items': items_list,
        'page': paginated_items.page,
        'per_page': paginated_items.per_page,
        'total_items': paginated_items.total,
        'total_pages': paginated_items.pages,
        'total_columns': total_columns,
        'total_rows': total_item_count 
    }
    return result

def get_table_as_dataframe(table_name):
    # Reflect the table using SQLAlchemy
    table = db.Table(table_name, db.metadata, autoload_with=db.engine)
    
    # Create a SQLAlchemy SELECT query
    query = db.select([table])

    # Execute the query using the db session and fetch all results
    result = db.session.execute(query)
    records = result.fetchall()

    # Create a list of dictionaries from the query result
    records_dict = [dict(record) for record in records]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(records_dict)

    return df

def get_table_as_dataframe_with_specific_columns(table_name, columns=None):
    # Reflect the table using SQLAlchemy
    table = Table(table_name, db.metadata, autoload_with=db.engine)
    
    # Check if specific columns are requested
    if columns:
        # Ensure the columns are valid table columns
        columns = [getattr(table.c, column) for column in columns if hasattr(table.c, column)]
    else:
        # Select all columns if none are specified
        columns = [table]
    
    # Create a SQLAlchemy SELECT query with specified columns
    query = db.select(columns)
    
    # Execute the query using the db session and fetch all results
    result = db.session.execute(query)
    records = result.fetchall()

    # Create a list of dictionaries from the query result
    records_dict = [dict(record) for record in records]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(records_dict)

    return df

def get_tables_as_dataframe(table_names, common_id_field):
    # Reflect all tables using SQLAlchemy
    tables = [db.Table(table_name, db.metadata, autoload_with=db.engine) for table_name in table_names]

    # Create SQLAlchemy SELECT queries for all tables
    queries = [db.select([table]) for table in tables]

    # Execute queries and fetch all results
    results = [db.session.execute(query).fetchall() for query in queries]

    # Create a list of dictionaries from the query results for each table
    records_dicts = [[dict(record) for record in records] for records in results]

    # Create DataFrames from the list of dictionaries for each table
    dataframes = [pd.DataFrame(records_dict) for records_dict in records_dicts]

    # Merge DataFrames based on the common ID field
    merged_df = pd.merge(dataframes[0], dataframes[1], on=common_id_field, how='outer')
    for df in dataframes[2:]:
        merged_df = pd.merge(merged_df, df, on=common_id_field, how='outer')

    return merged_df

class PaginatedQuery(Query):
    def paginate(self, page, per_page=10, error_out=True):
        if error_out and page < 1:
            abort(404)

        items = self.limit(per_page).offset((page - 1) * per_page).all()
        total = self.order_by(None).count()

        return {'items': items, 'total': total, 'page': page, 'per_page': per_page}

# Apply the PaginatedQuery to the SQLAlchemy session
db.session.query_class = PaginatedQuery

def get_table_as_dataframe_exception(table_name, filter_column=None, filter_value=None, page=1, per_page=10, distinct_column=None):
    page = int(page)
    try:
        per_page = int(per_page)
        # Reflect the table using SQLAlchemy
        table = db.Table(table_name, db.metadata, autoload_with=db.engine)

        # Create a SQLAlchemy SELECT query
        query = select([table])

        # Add a filter condition if provided
        if filter_column and filter_value:
            query = query.where(getattr(table.columns, filter_column) == filter_value)

        # Execute the query using the db session
        result = db.session.execute(query.limit(per_page).offset((page - 1) * per_page))

        # Fetch the paginated result as a list
        paginated_data = result.fetchall()

        # Convert the paginated result to a DataFrame
        df = pd.DataFrame(paginated_data, columns=result.keys())
        
        # Identify date columns
        df['updated_at_readable'] = pd.to_datetime(df['updated_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['created_at_readable'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        date_columns = df.select_dtypes(include=['datetime64[ns]']).columns

        # Drop date columns
        df = df.drop(columns=date_columns)
        df = df.where(pd.notnull(df), "")

        if filter_column and filter_value:
            total_rows_query = select([func.count()]).select_from(table).where(getattr(table.columns, filter_column) == filter_value)
        else:
            total_rows_query = select([func.count()]).select_from(table)
        # Apply specific filter for 'uniprot' table
        if table_name == 'membrane_protein_uniprot':
            total_rows_query = total_rows_query.where(~getattr(table.c, 'uniprot_id').like('%uniParcId:%'))
        
        # Ensure records are distinct based on pdb_code
        if distinct_column:
            # Select the distinct count of pdb_code
            total_rows_query = total_rows_query.with_only_columns(
                [func.count(func.distinct(getattr(table.c, "pdb_code")))]
            )

            # Print the compiled SQL query
            # print(str(total_rows_query.compile(compile_kwargs={"literal_binds": True})))
        total_rows = db.session.execute(total_rows_query).scalar()

        return {'data': df.to_dict(orient='records'), 'total_rows': total_rows, 'page': page, 'per_page': per_page}
    
    except SQLAlchemyError as e:
        # Handle SQLAlchemy-specific errors
        print({'error': str(e)})
        return {'data': [], 'total_rows': 0, 'page': 10, 'per_page': 10}
    except ValueError as e:
        # Handle validation errors
        print({'error': str(e)})
        return {'data': [], 'total_rows': 0, 'page': 10, 'per_page': 10}
    except Exception as e:
        # Handle other unexpected errors
        print({'error': 'An unexpected error occurred: ' + str(e)})
        return {'data': [], 'total_rows': 0, 'page': 10, 'per_page': 10}
    

def get_table_as_dataframe_download(table_name, columns=[], filter_column=None, filter_value=None):
    # Reflect the table using SQLAlchemy
    table = db.Table(table_name, db.metadata, autoload_with=db.engine)

    # Create a SQLAlchemy SELECT query
    if columns:
        select_columns = []
        for column in columns:
            if column.endswith('.*'):
                sub_table_name = column.split('.')[0]
                sub_table = db.Table(sub_table_name, db.metadata, autoload_with=db.engine)
                select_columns.extend(sub_table.columns)
            elif ' as ' in column:
                column_name, alias = column.split(' as ')
                column_obj = getattr(table.columns, column_name).label(alias)
                select_columns.append(column_obj)
            elif '.' in column:
                table_name, column_name = column.split('.')
                column_obj = getattr(db.Table(table_name, db.metadata, autoload_with=db.engine).columns, column_name)
                select_columns.append(column_obj)
            else:
                if hasattr(table.columns, column):
                    select_columns.append(getattr(table.columns, column))
        query = select(select_columns)
    else:
        query = select([table])

    # Add a filter condition if provided
    if filter_column and filter_value:
        if isinstance(filter_value, (list, tuple, set,)):
            query = query.where(getattr(table.columns, filter_column).in_(filter_value))
        else:
            query = query.where(getattr(table.columns, filter_column) == filter_value)

    # Execute the query using the db session
    result = db.session.execute(query)

    # Fetch all the data
    all_data = result.fetchall()

    # Convert the result to a DataFrame
    df = pd.DataFrame(all_data, columns=result.keys())

    # Calculate the total_rows separately
    total_rows = None
    if not filter_column and not filter_value:
        # Execute a count query without any filter condition
        total_rows = db.session.execute(select([func.count()]).select_from(table)).scalar()
    elif filter_column:
        total_rows = db.session.execute(select([func.count()]).select_from(table).where(getattr(table.columns, filter_column) == filter_value)).scalar()

    return {'data': df, 'total_rows': total_rows}

def export_to_csv(df, csv_filename):
    df.to_csv(csv_filename, index=False)

def export_to_excel(df, excel_filename):
    df.to_excel(excel_filename, index=False)
     
def getMPstructDB():
    table_df_MPstruct = get_table_as_dataframe("membrane_protein_mpstruct")
    return table_df_MPstruct.columns.tolist()
    
def getPDBDB():
    table_df_DB = get_table_as_dataframe("membrane_protein_pdb")
    return table_df_DB.columns.tolist()

def getOPMDB():
    table_df_MPstruct = get_table_as_dataframe("membrane_protein_opm")
    return table_df_MPstruct.columns.tolist()
    
def getUniprotDB():
    table_df_DB = get_table_as_dataframe("membrane_protein_uniprot")
    return table_df_DB.columns.tolist()

def preprocessVariables(variables:list=[]):
    formatted_strings = []

    for string in variables:
        # Split the string by underscores
        words = string.split('_')

        # Remove the first word if the number of words is greater than 4
        if len(words) > 4:
            words = words[1:]

        # Capitalize each remaining word and join them back into a string
        formatted_string = ' '.join(word.capitalize() for word in words)
        formatted_strings.append(formatted_string)

    return formatted_strings

def all_merged_databases():
    table_names = ['membrane_proteins', 'membrane_protein_opm']
    result_df = get_tables_as_dataframe(table_names, "pdb_code")
    result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")

    # find columns in common other than the key
    common = set(result_df.columns) - {"pdb_code"} & set(result_df_uniprot.columns)

    # drop them from the “right” frame
    right_pruned = result_df_uniprot.drop(columns=list(common))

    all_data = pd.merge(right=result_df, left=right_pruned, on="pdb_code", how="outer")
    
    return all_data
    
def search_merged_databases(pdb_code):
    all_data = all_merged_databases()
    # all_data = all_data.where(pd.notnull(all_data), "")
    data = all_data[(all_data["pdb_code"].fillna('').str.upper() == pdb_code.upper()) | (all_data["uniprot_id"].fillna('').str.upper() == pdb_code.upper())].to_dict(orient='records')

    # Clean NaN for display
    for record in data:
        for k, v in record.items():
            if pd.isna(v):
                record[k] = None

    return data


def get_columns_by_pdb_codes(pdb_codes, columns):
    """
    Given a list of pdb_codes and a list of column names, 
    returns those columns (plus pdb_code) for the matching rows.
    """
    # 1) pull in the full merged DataFrame
    df = all_merged_databases()

    # 2) normalize & filter by pdb_code (case-insensitive)
    upc = [code.upper() for code in pdb_codes]
    df = df[df["pdb_code"].fillna("").str.upper().isin(upc)]

    # 3) validate requested columns
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Columns not found in merged DB: {missing}")

    # 4) select only pdb_code + your columns, fill NaNs, and return
    cols = ["pdb_code"] + columns
    return df[cols].fillna("").to_dict(orient="records")




######################################## LIST OF OPTION FOR FILTERS ################################
#                                                                                                  #
#                                                                                                  #
####################################################################################################
cache = RedisCache()

def group_filter_options():
    cache_key = "unique_group"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_proteins", ["group"])
    option_lists = option_lists_frame['group'].unique()
    
    # Cache the data before returning
    # transform to list first
    option_lists = ['All'] + option_lists.tolist()
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def subgroup_filter_options():
    cache_key = "unique_subgroup"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_proteins", ["subgroup"])
    option_lists = option_lists_frame['subgroup'].unique()
    
    # Cache the data before returning
    # transform to list first
    option_lists = ['All'] + option_lists.tolist()
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def taxonomic_domain_filter_options():
    cache_key = "unique_taxonomic_domain"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_proteins", ["taxonomic_domain"])
    option_lists = option_lists_frame['taxonomic_domain'].unique()
    
    # Cache the data before returning
    # transform to list first
    option_lists = ['All'] + option_lists.tolist()
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def experimental_methods_filter_options():
    cache_key = "unique_experimental_methods"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_proteins", ["rcsentinfo_experimental_method"])
    option_lists = option_lists_frame['rcsentinfo_experimental_method'].unique()
    
    # Cache the data before returning
    # transform to list first
    option_lists = ['All'] + option_lists.tolist()
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def molecular_function_filter_options():
    cache_key = "unique_molecular_function"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_protein_uniprot", ["molecular_function"])
    
    # Split the 'activities' column into lists
    option_lists_frame['molecular_function_list'] = option_lists_frame['molecular_function'].str.split(';')

    # Flatten the list of lists into a single list and strip whitespace
    all_molecular_function = [molecular_function.strip() for sublist in option_lists_frame['molecular_function_list'] for molecular_function in sublist]

    # Get unique activities
    option_lists = ['All'] + list(set(all_molecular_function))
    
    # Cache the data before returning
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def cellular_component_filter_options():
    cache_key = "unique_cellular_component"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_protein_uniprot", ["cellular_component"])
    
    # Split the 'activities' column into lists
    option_lists_frame['cellular_component_list'] = option_lists_frame['cellular_component'].str.split(';')

    # Flatten the list of lists into a single list and strip whitespace
    all_cellular_component = [cellular_component.strip() for sublist in option_lists_frame['cellular_component_list'] for cellular_component in sublist]

    # Get unique activities
    option_lists = ['All'] + list(set(all_cellular_component))
    
    # Cache the data before returning
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def biological_process_filter_options():
    cache_key = "unique_biological_process"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_protein_uniprot", ["biological_process"])
    # Split the 'activities' column into lists
    option_lists_frame['biological_process_list'] = option_lists_frame['biological_process'].str.split(';')

    # Flatten the list of lists into a single list and strip whitespace
    all_biological_process = [biological_process.strip() for sublist in option_lists_frame['biological_process_list'] for biological_process in sublist]

    # Get unique activities
    option_lists = ['All'] + list(set(all_biological_process))
    
    # Cache the data before returning
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def family_name_filter_options():
    cache_key = "unique_family_names"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_protein_opm", ["family_name_cache"])
    option_lists = option_lists_frame['family_name_cache'].unique()
    
    # Cache the data before returning
    # transform to list first
    option_lists = ['All'] + option_lists.tolist()
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def species_filter_options():
    cache_key = "unique_species"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_protein_opm", ["species_name_cache"])
    option_lists = option_lists_frame['species_name_cache'].unique()
    
    # Cache the data before returning
    # transform to list first
    option_lists = ['All'] + option_lists.tolist()
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def membrane_name_filter_options():
    cache_key = "unique_membrane_name"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_protein_opm", ["membrane_name_cache"])
    option_lists = option_lists_frame['membrane_name_cache'].unique()
    
    # Cache the data before returning
    # transform to list first
    option_lists = ['All'] + option_lists.tolist()
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def super_family_filter_options():
    cache_key = "unique_super_family_names"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_protein_opm", ["family_superfamily_name"])
    option_lists = option_lists_frame['family_superfamily_name'].unique()
    # Cache the data before returning
    # transform to list first
    option_lists = ['All'] + option_lists.tolist()
    
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists

def super_family_class_type_filter_options():
    cache_key = "unique_family_superfamily_classtype_names"
    
    # Check if the data is already cached
    cached_data = cache.get_item(cache_key)
    if cached_data is not None:
        return cached_data
    
    # If not cached, fetch from the database
    option_lists_frame = get_table_as_dataframe_with_specific_columns("membrane_protein_opm", ["family_superfamily_classtype_name"])
    option_lists = option_lists_frame['family_superfamily_classtype_name'].unique()
    
    # Cache the data before returning
    # transform to list first
    option_lists = ['All'] + option_lists.tolist()
    ttl_in_seconds = timedelta(days=10).total_seconds()
    cache.set_item(cache_key, option_lists, ttl=ttl_in_seconds)  # Cache for 1 hour
    return option_lists


def convert_to_list_of_dicts(input_str):
    # Split the input string by comma and strip any extra spaces
    keys = [key.strip() for key in input_str.split(",")]
    
    # Create a set to store unique dictionaries
    unique_dicts = set()
    
    for key in keys:
        # Create a dictionary entry
        entry = {
            "key": key,
            "name": key,
            "x-angle": 5 if key == "group" else 0,
        }
        
        # Convert the dictionary to a tuple of tuples to store in the set
        entry_tuple = tuple(entry.items())
        
        # Add the tuple to the set to ensure uniqueness
        unique_dicts.add(entry_tuple)
    
    # Convert the set of tuples back to a list of dictionaries
    result = [dict(t) for t in unique_dicts]
    
    return result


import matplotlib.colors as mcolors
def generate_color_palette(start_color, end_color, num_colors):
    # Convert hex colors to RGB
    start_rgb = mcolors.hex2color(start_color)
    end_rgb = mcolors.hex2color(end_color)

    # Create a list of RGB colors in the gradient
    colors = []
    for i in range(num_colors):
        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * (i / (num_colors - 1))
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * (i / (num_colors - 1))
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * (i / (num_colors - 1))
        colors.append((r, g, b))

    # Convert RGB colors back to hex
    hex_colors = [mcolors.rgb2hex(color) for color in colors]

    return hex_colors


def create_grouped_bar_chart(table_df):
    
    # Identify date columns
    date_columns = table_df.select_dtypes(include=['datetime64[ns]']).columns
    
    # Drop date columns
    table_df = table_df.drop(columns=date_columns)
    
    # Group by 'group' and count the occurrences
    grouped_data = table_df.groupby("group").size().reset_index(name='CumulativeCount')
    
    # Sort by count
    grouped_data = grouped_data.sort_values(by='CumulativeCount', ascending=True)
    grouped_data['group'] = grouped_data['group'].replace({
        'MONOTOPIC MEMBRANE PROTEINS': "Group 1",
        'TRANSMEMBRANE PROTEINS:BETA-BARREL': "Group 2",
        'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL': "Group 3",
    })
    
    # Define group labels and their meanings
    group_labels = {
        "Group 1": 'Group 1 (MONOTOPIC MEMBRANE PROTEINS)',
        "Group 2": 'Group 2 (TRANSMEMBRANE PROTEINS:BETA-BARREL)',
        "Group 3": 'Group 3 (TRANSMEMBRANE PROTEINS:ALPHA-HELICAL)'
    }
    
    # Add a 'label' column for detailed legend information
    grouped_data['label'] = grouped_data['group'].map(group_labels)
    
    # Define color list
    color_list = ['#D9DE84', '#93C4F6', '#005EB8', '#636B05']
    
    # Create the grouped bar chart
    chart = alt.Chart(grouped_data).mark_bar().encode(
        x=alt.X(
            'group:N', title='Group', 
            sort=None, axis=alt.Axis(
                labelAngle=0,
                labelLimit=0
            )
        ),
        y=alt.Y('CumulativeCount:Q', title='Cumulative MP Structures'),
        color=alt.Color(
            'label:N', scale=alt.Scale(domain=list(group_labels.values()), range=color_list),
            legend=alt.Legend(title="Group", orient="bottom", labelLimit=0, direction="vertical")
        ),
        tooltip=["group", "CumulativeCount"]
    ).properties(
        title='Cumulative sum of resolved Membrane Protein (MP) Structures categorized by group',
        width="container"
    ).configure_legend(
        symbolType='square'
    ).configure_axisX(
        labelAngle=0  # Ensure labels are horizontal
    )

    return convert_chart(chart)


def extract_widths(chart_dict, chart_width=800):
    widths = []
    
    # Calculate the available width by subtracting the padding
    padding = 200 
    available_width = chart_width - padding

    chart_width_1 = 0.5*available_width
    chart_width_2 = 0.5*chart_width_1
    new_widths = [chart_width_2, chart_width_2, chart_width_1]
    # Check hconcat items
    if 'hconcat' in chart_dict:
        for item in chart_dict['hconcat']:
            width = item.get('width')
            if width:
                widths.append(width)
            if 'layer' in item:
                for layer in item['layer']:
                    width = layer.get('width')
                    if width:
                        widths.append(width)

    return widths == new_widths