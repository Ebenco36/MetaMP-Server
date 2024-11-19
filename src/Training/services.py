# services.py
import pandas as pd
import altair as alt
from src.Dashboard.data import (
    EM_columns, MM_columns, 
    NMR_columns, X_ray_columns
)
from datetime import datetime
from sqlalchemy.orm import joinedload
from src.MP.services import DataService
from altair.vegalite.v5.api import Chart
from sqlalchemy.exc import IntegrityError
from src.MP.model import MembraneProteinData
from src.Training.models import (
    FilterToolOption, db, Category, 
    Question, Option, FilterTool, UserResponse
)
from src.Dashboard.services import get_table_as_dataframe
from src.Jobs.Utils import separate_numerical_categorical
from src.MP.machine_learning_services import OutlierDetection
from src.Jobs.transformData import report_and_clean_missing_values


class CategoryService:
    @staticmethod
    def get_category(category_id):
        return Category.query.get_or_404(category_id)

    @staticmethod
    def delete_category(category_id):
        category = Category.query.get_or_404(category_id)
        if not category.questions:
            db.session.delete(category)
            db.session.commit()
            return True
        return False

    @staticmethod
    def get_all_categories():
        result = (
            Category.query
            .options(
                joinedload(Category.questions)
                .joinedload(Question.options)
            )
            .filter(Category.questions.any(Question.options.any()))
            .order_by(Category.id.asc())  # Order categories by ID (you can change this)
            .all()
        )

        # Order questions by item_order within each category
        for category in result:
            category.questions.sort(key=lambda q: q.item_order)
        return result

    @staticmethod
    def create_category(name, description):
        existing_category = Category.query.filter_by(name=name).first()
        if existing_category:
            return None
        new_category = Category(name=name, description=description)
        db.session.add(new_category)
        db.session.commit()
        return new_category

    @staticmethod
    def update_category(category_id, name, description):
        category = Category.query.get_or_404(category_id)

        # Check if a category with the new name already exists
        existing_category = Category.query.filter(
            Category.id != category_id,
            Category.name == name,
        ).first()

        if existing_category:
            return None  # Another category with the same name already exists

        category.name = name
        category.description = description

        try:
            db.session.commit()
            return category
        except IntegrityError:
            db.session.rollback()
            return None  # IntegrityError indicates a unique constraint violation

class QuestionService:
    @staticmethod
    def create_question(text, category_id, item_order, question_type, instruction, hints):
        question = Question(
            text=text,
            category_id=category_id,
            item_order=item_order,
            question_type=question_type,
            instruction=instruction,
            hints=hints,
        )
        db.session.add(question)
        db.session.commit()
        return question

    @staticmethod
    def get_all_questions():
        return Question.query.all()
    
    @staticmethod
    def get_question_by_id(question_id):
        return Question.query.get(question_id)

    @staticmethod
    def update_question(question_id, text, category_id, item_order, question_type, instruction, hints):
        question = Question.query.get_or_404(question_id)
        question.text = text
        question.category_id = category_id
        question.item_order = item_order
        question.question_type = question_type
        question.instruction = instruction
        question.hints = hints
        db.session.commit()

    @staticmethod
    def delete_question(question_id):
        question = Question.query.get_or_404(question_id)
        db.session.delete(question)
        db.session.commit()

class OptionService:
    @staticmethod
    def create_option(text, question_id, is_correct=False):
        option = Option(text=text, question_id=question_id, is_correct=is_correct)
        db.session.add(option)
        db.session.commit()
        return option

    @staticmethod
    def get_all_options():
        return Option.query.all()

    @staticmethod
    def get_option_by_id(option_id):
        return Option.query.get(option_id)

    @staticmethod
    def update_option(option_id, text, is_correct):
        option = Option.query.get_or_404(option_id)
        option.text = text
        option.is_correct = is_correct
        db.session.commit()

    @staticmethod
    def delete_option(option_id):
        option = Option.query.get_or_404(option_id)
        db.session.delete(option)
        db.session.commit()

# FilterToolService
class FilterToolService:
    @staticmethod
    def create_filter_tool(title, name, parent, selected_option, question_id, options_data):
        filter_tool = FilterTool(
            title=title,
            name=name,
            parent = parent,
            selected_option=selected_option,
            question_id=question_id
        )
        db.session.add(filter_tool)
        db.session.flush()

        # Create options for the filter tool
        options = [
            FilterToolOption(text=option['text'], value=option['value'], filter_tool=filter_tool)
            for option in options_data
        ]
        db.session.add_all(options)
        db.session.commit()

        return filter_tool

    @staticmethod
    def get_filter_tool(filter_tool_id):
        return FilterTool.query.get(filter_tool_id)

    @staticmethod
    def update_filter_tool(filter_tool_id, title, name, parent, selected_option, options_data):
        filter_tool = FilterTool.query.options(db.joinedload(FilterTool.options)).get_or_404(filter_tool_id)
        filter_tool.title = title
        filter_tool.name = name
        filter_tool.parent = parent
        filter_tool.selected_option = selected_option

        # Update options for the filter tool
        for option_data, filter_tool_option in zip(options_data, filter_tool.options):
            filter_tool_option.text = option_data['text']
            filter_tool_option.value = option_data['value']

        db.session.commit()

    @staticmethod
    def delete_filter_tool(filter_tool_id):
        filter_tool = FilterTool.query.get_or_404(filter_tool_id)
        db.session.delete(filter_tool)
        db.session.commit()

class UserResponseService:
    @staticmethod
    def get_user_responses(user_id):
        return UserResponse.query.filter_by(user_id=user_id).all()

    @staticmethod
    def create_user_response(user_id, question_id, option_id):
        user_response = UserResponse(user_id=user_id, question_id=question_id, option_id=option_id)
        db.session.add(user_response)
        db.session.commit()
        return user_response

    @staticmethod
    def check_user_response(user_id, session_id, data_list):
        # Create records in the UserResponse model
        for index, data_item in enumerate(data_list.get("answers")):
            # Check if a response already exists for the user and question
            existing_response = UserResponse.query.filter_by(
                session_id=session_id,
                user_id=user_id,
                question_id=int(data_item['question_id'])
            ).first()

            if existing_response:
                # Update the existing response
                existing_response.answer_id = data_item['id']
                existing_response.is_correct = data_item['is_correct']
                existing_response.duration = data_list.get("time")['duration']
                existing_response.endTime = data_list.get("time")['endTime']
                existing_response.startTime = data_list.get("time")['startTime']
                existing_response.time_taken = data_list.get("questionBasedTimer")[index].get(str(index+1))
                existing_response.updated_at = datetime.utcnow()
            else:
                # Create a new response
                user_response = UserResponse(
                    session_id=session_id,
                    user_id=user_id,
                    question_id=data_item['question_id'],
                    answer_id=data_item['id'],
                    is_correct=data_item['is_correct'],
                    duration = data_list.get("time")['duration'],
                    endTime = data_list.get("time")['endTime'],
                    startTime = data_list.get("time")['startTime'],
                    time_taken = data_list.get("questionBasedTimer")[index].get(str(index+1)),
                )
                db.session.add(user_response)

        # Commit the changes
        db.session.commit()

        return {'message': "Added or updated user responses"}

    
    
    
    """
        implement classes based on the different methods
        There is a lot here..
        Current methods are: X-ray, 
    """
        
def get_records_on_method():
    query = MembraneProteinData.query.filter_by(rcsentinfo_experimental_method="NMR")
    sql_statement = str(query.statement)
    
    # Example usage with filter
    filter_column = 'rcsentinfo_experimental_method'
    filter_value = 'NMR'
    result_df = get_table_as_dataframe("membrane_proteins", filter_column, filter_value)
    return result_df

def get_quantification_data(key = "X-ray"):
    if(key == "X-ray"):
        return X_ray_columns()
    elif(key == "EM"):
        return EM_columns()
    elif(key == "NMR"):
        return NMR_columns()
    elif(key == "MM"):
        return MM_columns()
    else:
        return list(set(MM_columns()) & set(NMR_columns()) & set(EM_columns()) & set(X_ray_columns()))
    
def getChartForQuestion(column = "rcsentinfo_experimental_method", filter = "X-ray"):
    data = DataService.get_data_by_column_search_download(column, filter)
    data = data.get("data")
    """
    data = data.drop(columns = [
        "bibliography", "secondary_bibliogrpahies", "related_pdb_entries", "member_proteins",
        "audit_author", "is_replaced", "pdb_code_changed", "citation", "exptl", "pdbx_database_related", 
        "pdbx_audit_support", "em3d_reconstruction", "em3d_fitting", "em_ctf_correction", "em_entity_assembly", 
        "em_image_recording", "em_imaging", "em_particle_selection", "em_software", "em_specimen", "em_vitrification", 
        "rcsb_external_references", "pdbx_audit_revision_details", "pdbx_audit_revision_history", 
        "em_single_particle_entity", "created_at", "updated_at", "diffrn", "diffrn_radiation", "refine_details",
        "exptl_crystal", "pdbx_audit_revision_group", "refine", "refine_hist", "refine_ls_restr", "software",
        "diffrn_detector", "pdbx_audit_revision_category", "pdbx_audit_revision_item", "refine_analyze", 
        "reflns", "diffrn_source", "exptl_crystal_grow", "rcsb_binding_affinity", "reflns_shell", "expcrygrow_pdbx_details",
        "pdbx_initial_refinement_model", "pdbx_reflns_twin"
    ])
    """
    return data

def generate_option_ids(options):
    for i, option in enumerate(options, start=1):
        option["id"] = i  # Generate incrementing numerical IDs for each option

def generate_question_ids(questions):
    for i, question in enumerate(questions, start=1):
        question["id"] = i  # Generate incrementing numerical IDs for each question
        generate_option_ids(question["options"])

def filter_questions_in_sets(question_sets, class_name=""):
    filtered_question_sets = []

    for questions_set in question_sets:
        generate_question_ids(questions_set["questions"])
        filtered_questions = [question for question in questions_set["questions"] if question["question_type"]]
        filtered_set = {**questions_set, "questions": filtered_questions}
        filtered_question_sets.append(filtered_set)

    return filtered_question_sets




############################################################
############# Implementation for Inconsistencies ###########
############################################################

def transform_dataframe(data):
    """
    Transforms the dataframe by splitting and exploding list-like columns.
    
    Args:
    - data (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The transformed dataframe.
    """
    df = data.copy()
    df['pdb_codes'] = df['pdb_codes'].str.split(', ')
    df['protein_codes'] = df['pdb_codes']
    df['family_superfamily_classtype_name'] = df['family_superfamily_classtype_name'].str.split(', ')
    df['group'] = df['group'].str.split(', ')
    df['experimental_method'] = df['rcsentinfo_experimental_method'].str.split(', ')
    df['inconsistency'] = 1

    df = df.explode(['pdb_codes', 'family_superfamily_classtype_name', 'group', 'experimental_method'])
    df = df.rename(columns={
        'pdb_codes': 'pdb_code', 
        'family_superfamily_classtype_name': 'group (OPM)', 
        'group': 'group (MPstruc)'
    })
    df = df.reset_index(drop=True)
    
    df['experimental_method'] = df['experimental_method'].replace({
        'EM': 'Cryo-Electron Microscopy (EM)',
        'X-ray': 'X-Ray Crystallography',
        'NMR': 'Nuclear Magnetic Resonance (NMR)',
    })
    df.to_csv("inconsistencies_by_year.csv")
    return df

def find_inconsistencies(row):
    """
    Checks for inconsistencies in the given row based on keyword groups.
    
    Args:
    - row (pd.Series): The row to check.
    - keyword_groups (dict): A dictionary mapping keywords to expected group values.

    Returns:
    - bool: True if an inconsistency is found, False otherwise.
    """
    family_name = row['family_superfamily_classtype_name']
    group_value = row['group']
    
    # Check if the group value is different from the family name
    return group_value.lower() != family_name.lower()

def aggregate_inconsistencies(data):
    """
    Aggregates inconsistencies by year and collects PDB codes.
    
    Args:
    - data (pd.DataFrame): The input dataframe.
    - keyword_groups (dict): A dictionary mapping keywords to expected group values.

    Returns:
    - pd.DataFrame: The aggregated inconsistencies by year.
    """
    # Define a dictionary to map keywords in famsupclasstype_type_name to expected group values
    # expected_groups = {
    #     'Monotopic': 'MONOTOPIC MEMBRANE PROTEINS',
    #     'Transmembrane': 'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL',
    #     'Transmembrane': 'TRANSMEMBRANE PROTEINS:BETA-BARREL'
    #     # Add more mappings as needed
    # }
    look_up_table = {
        "Alpha-helical polytopic": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL",
        "Bitopic proteins": "BITOPIC PROTEINS",
        "Beta-barrel transmembrane": "TRANSMEMBRANE PROTEINS:BETA-BARREL",
        "All alpha monotopic/peripheral": "MONOTOPIC MEMBRANE PROTEINS",
        "All beta monotopic/peripheral": "MONOTOPIC MEMBRANE PROTEINS",
        "Alpha/Beta monotopic/peripheral": "MONOTOPIC MEMBRANE PROTEINS",
        "Alpha + Beta monotopic/peripheral": "MONOTOPIC MEMBRANE PROTEINS",
        "Alpha-helical peptides": "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL"
    }
    # Map family_superfamily_classtype_name to look-up table and handle missing records
    data['family_superfamily_classtype_name'] = data['family_superfamily_classtype_name'].map(look_up_table)

    data['inconsistency'] = data.apply(lambda row: find_inconsistencies(row), axis=1)
    # Aggregate the inconsistencies by year
    inconsistencies_by_year = data[data['inconsistency']].groupby('bibliography_year').agg(
        inconsistencies=('inconsistency', 'sum'),
        pdb_codes=('pdb_code', lambda x: ', '.join(x)),
        family_superfamily_classtype_name=('family_superfamily_classtype_name', lambda x: ', '.join(x)),
        group=('group', lambda x: ', '.join(x)),
        rcsentinfo_experimental_method=('rcsentinfo_experimental_method', lambda x: ', '.join(x))
    ).reset_index()
    
    return inconsistencies_by_year


def create_visualization(data, chart_width=None):
    """
    Creates a visualization of inconsistencies by year and protein type.

    Args:
    - data (pd.DataFrame): The input dataframe.
    - chart_width (int): The width of the chart and table.

    Returns:
    - alt.Chart: The combined Altair chart and table.
    """
    # Define selections
    brush = alt.selection_interval(encodings=["x", "y"])
    click = alt.selection_point(fields=['inconsistencies'], name='click')

    # Check and set chart width
    if chart_width and isinstance(chart_width,int):
        chart_width = int(chart_width) - 50
    else:
        chart_width = "container"

    # Create line chart
    line_chart = alt.Chart(data).mark_line(point=True, interpolate='monotone').encode(
        x=alt.X('bibliography_year:O', title="Year"),
        y=alt.Y(
            'inconsistencies:Q', 
            title="Inconsistencies",
            scale=alt.Scale(domain=(0, data['inconsistencies'].max() * 1.1))  # Added a small buffer to y-axis
        ),
        tooltip=[
            'bibliography_year', 'inconsistencies', 
            'protein_codes', 'group (OPM)', 
            'group (MPstruc)'
        ]
    ).add_params(
        brush, click
    ).properties(
        width=chart_width,
        title='Discrepancies in membrane protein structure groups observed over time using the OPM and MPstruc databases.',
    )
    
    # Create table
    table = alt.Chart(data).mark_text(align='left').encode(
        y=alt.Y('row_number:O', axis=None),
    ).transform_filter(
        brush
    ).transform_filter(
        click
    ).transform_window(
        row_number='row_number()'
    ).transform_filter(
        'datum.row_number < 15'
    )

    width_array = [80, 80, 100, 80, 120]
    
    # Create individual columns
    pdb_code = table.encode(
        text='pdb_code:N'
    ).properties(
        width=width_array[0],
        title=alt.TitleParams(text='PDB Code', align='left')
    )

    group = table.encode(
        text='group (MPstruc):N'
    ).properties(
        width=width_array[1],
        title=alt.TitleParams(text='Group (MPstruc)', align='left')
    )

    OPM_group = table.encode(
        text='group (OPM):N'
    ).properties(
        width=width_array[2],
        title=alt.TitleParams(text='Group (OPM)', align='left')
    )

    year = table.encode(
        text='bibliography_year:N'
    ).properties(
        width=width_array[3],
        title=alt.TitleParams(text='Year', align='left')
    )

    method = table.encode(
        text='experimental_method:N'
    ).properties(
        width=width_array[4],
        title=alt.TitleParams(text='Experimental Method', align='left')
    )
    
    # Concatenate columns horizontally
    table_layout = alt.hconcat(
        pdb_code, group, OPM_group, year, method
    ).resolve_legend(
        color="independent"
    )

    # Concatenate chart and table vertically
    chart_with_table = alt.vconcat(
        line_chart, table_layout
    ).configure_view(
        strokeWidth=0
    )

    return chart_with_table


def create_visualizationxxxxx(data, chart_width=None):
    """
    Creates a visualization of inconsistencies by year and protein type.

    Args:
    - data (pd.DataFrame): The input dataframe.

    Returns:
    - alt.Chart: The combined Altair chart and table.
    """
    # Define selections
    brush = alt.selection_interval(encodings=["x", "y"])
    click = alt.selection_point(fields=['inconsistencies'], name='click')

    # Check and set chart width
    if chart_width and isinstance(chart_width,int):
        chart_width = int(chart_width) - 50
    else:
        chart_width = "container"

    # Create line chart
    line_chart = alt.Chart(data).mark_line(point=True, interpolate='monotone').encode(
        x=alt.X('bibliography_year:O', title="Year"),
        y=alt.Y(
            'inconsistencies:Q', 
            title="Inconsistencies",
            scale=alt.Scale(domain=(0, data['inconsistencies'].max() * 1.1))  # Added a small buffer to y-axis
        ),
        tooltip=[
            'bibliography_year', 'inconsistencies', 
            'protein_codes', 'group (OPM)', 
            'group (MPstruc)'
        ]
    ).add_params(
        brush, click
    ).properties(
        width=chart_width,
        title='Discrepancies in membrane protein structure groups observed over time using the OPM and MPstruc databases.',
    )
    

    # Determine table width and column width
    if chart_width != "container":
        table_width = chart_width
        column_width = table_width / 5
    else:
        table_width = "container"
        column_width = 100  # Default column width in container mode

    # Create table
    table = alt.Chart(data).mark_text(align='left').encode(
        y=alt.Y('row_number:O', axis=None),
    ).transform_filter(
        brush
    ).transform_filter(
        click
    ).transform_window(
        row_number='row_number()'
    ).transform_filter(
        'datum.row_number < 15'
    ).properties(
        #width=table_width
    )

    width_array = [30, 30, 40, 30, 40]
    
    # Create individual columns
    pdb_code = table.encode(
        text='pdb_code:N'
    ).properties(
        width=width_array[0], # column_width,
        title=alt.TitleParams(text='PDB Code', align='left')
    )

    group = table.encode(
        text='group (MPstruc):N'
    ).properties(
        width=width_array[1], #column_width,
        title=alt.TitleParams(text='Group (MPstruc)', align='left')
    )

    OPM_group = table.encode(
        text='group (OPM):N'
    ).properties(
        width=width_array[2], #column_width,
        title=alt.TitleParams(text='Group (OPM)', align='left')
    )

    year = table.encode(
        text='bibliography_year:N'
    ).properties(
        width=width_array[3], #column_width,
        title=alt.TitleParams(text='Years', align='left')
    )

    method = table.encode(
        text='experimental_method:N'
    ).properties(
        width=width_array[4], #column_width,
        title=alt.TitleParams(text='Experimental Method', align='left')
    )
    

    # Concatenate columns horizontally
    table_layout = alt.hconcat(
        pdb_code, year, group, OPM_group, method
    ).resolve_legend(
        color="independent"
    )

    # Concatenate chart and table vertically
    chart_with_table = alt.vconcat(
        line_chart, table_layout
    ).configure_view(
        strokeWidth=0
    )

    return chart_with_table
 
def group_annotation(chart_obj:Chart, group_list=[1, 2, 3]):
    data = {
        "Group": group_list,
        "Full Text": [
            "MONOTOPIC MEMBRANE PROTEINS", 
            "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL", 
            "TRANSMEMBRANE PROTEINS:BETA-BARREL"
        ]
    }
    source = pd.DataFrame(data)
    
    # Base chart for data tables
    ranked_text = alt.Chart(source).mark_text().encode(
        y=alt.Y('row_number:O',axis=None)
    ).transform_window(
        row_number='row_number()'
    ).transform_window(
        rank='rank(row_number)'
    ).transform_filter(
        alt.datum.rank<20
    )

    # Data Tables
    keys = ranked_text.encode(text='Group:N').properties(title=alt.TitleParams(text='Group', align='center'), width=20)
    full_text = ranked_text.encode(text='Full Text:N').properties(title=alt.TitleParams(text='Full Text', align='center'), width=20)
    text = alt.hconcat(keys, full_text).properties(title="Annotations") # Combine data tables

    # Build chart
    return alt.vconcat(
        chart_obj,
        text
    ).configure_axis(
        labelLimit=0,
    ).resolve_legend(
        color="independent"
    ).configure_view(
        strokeWidth=0
    )
    
    
########################################################
############# End Of Data Inconsistencies ##############
########################################################
from utils.package import evaluate_dimensionality_reduction

def clean_column_name(col):
    if col.startswith('rcsentinfo_'):
        col = col[len('rcsentinfo_'):]  # Remove prefi
        
    elif col.startswith('em3d_'):
        col = col[len('em3d_'):]
    
    elif col.startswith('em_'):
        col = col[len('em_'):]
        
    elif col.startswith('exptl_'):
        col = col[len('exptl_'):]
    
    return col

def prepare_data(numerical_data, categorical_data, selected_columns):
    categorical_data = categorical_data.reset_index(drop=True)
    numerical_data = numerical_data.reset_index(drop=True)
    new_df = pd.concat([numerical_data, categorical_data], axis=1).dropna().reset_index(drop=True)
    return new_df[selected_columns + categorical_data.columns.to_list()]

def create_pairwise_plot(
    data=None, selected_columns=None, 
    categorical_columns=None, width_chart_single=None, 
    width_outlier_detection=None, 
    plot_attrs=[], 
    axis_name="PCA",
    outlier_chart_title=""
):
    reverse_input = selected_columns[::-1]
    brush = alt.selection_interval(resolve='global')

    pairwise_plot = alt.Chart(data).mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative', axis=alt.Axis(format='~s')),
        alt.Y(alt.repeat("row"), type='quantitative', axis=alt.Axis(format='~s')),
        color=alt.condition(
            brush,
            alt.value('#4c78a8'),  # Remove legend when the condition is met
            alt.value('lightgray')  # Default color
        ),
        tooltip=categorical_columns
    ).add_params(
        brush
    ).properties(
        width=width_chart_single,
        height=width_chart_single
    ).repeat(
        row=selected_columns,
        column=reverse_input
    ).properties(
        title="Scatter Plot Matrix (SPLOM) for selected attributes"
    )
    
    outlier_chart = alt.Chart(data).mark_circle().encode(
        x=alt.X(plot_attrs[0], title=axis_name + " 1"),
        y=alt.X(plot_attrs[1], title=axis_name + " 2"),
        color=alt.condition(
            brush,
            alt.Color('Outlier:N', legend=alt.Legend(
                orient="bottom",
                columns=5, 
                columnPadding=20, 
                labelLimit=0, 
                direction = 'vertical'
            )),
            alt.value('lightgray'),
        ),
        tooltip=categorical_columns
    ).add_params(
        brush
    ).properties(
        title=outlier_chart_title,
        width=width_outlier_detection
    )
        

    return outlier_chart, pairwise_plot

def calculate_pearson_correlation(data, selected_columns):
    return data[selected_columns].corr(method='pearson')

def perform_dimensionality_reduction(numerical_data, methods_params):
    reduced_data, plot_data = evaluate_dimensionality_reduction(numerical_data, methods_params)
    return pd.concat(plot_data)

def merge_reduced_data_with_categorical(reduced_data, numerical_data, categorical_data):
    combined_data = []
    for method in ['PCA', 't-SNE', 'UMAP']:
        method_data = reduced_data[reduced_data["Method"] == method].reset_index(drop=True)
        combined_method_data = pd.concat([method_data, numerical_data, categorical_data], axis=1)
        combined_data.append(combined_method_data)
    return combined_data

def detect_and_visualize_outliers(data, method, detector, axis_name="PCA", width_chart_single=None):
    detected_data = detector.fit_predict(data, method)
    chart = detector.visualize(
        f'Outlier Detection using {method}', 
        data.columns.to_list(), 
        axis_name=axis_name, 
        width_chart_single=width_chart_single
    )
    return chart

def detect_outliers_and_data(data, method, detector):
    detected_data = detector.fit_predict(data, method)
    return detected_data

########################################################
############# Outlier Detection Section ################
########################################################


def filter_data_by_method(data, method="X-ray"):
    return data[data["rcsentinfo_experimental_method"] == method]

def clean_data(data, threshold=30):
    return report_and_clean_missing_values(data, threshold)

def separate_columns(data):
    numerical_cols, categorical_cols = separate_numerical_categorical(data)
    numerical_data = data[numerical_cols]
    categorical_data = data[categorical_cols][[
        "group", "subgroup", 
        "rcsentinfo_experimental_method", 
        "pdb_code"
    ]]
    return numerical_data, categorical_data

def identify_columns_to_drop(numerical_data):
    dynamic_columns_to_drop = [
        col for col in numerical_data.columns 
        if '_count_' in col or col.startswith('count_') or col.endswith('_count') or
           col.startswith('revision_') or col.endswith('_revision') or
           col.startswith('id_') or col.endswith('_id') or col == "id"
    ]
    additional_columns_to_drop = [
        "index", "bibliography_year", "cell_angle_alpha", "cell_angle_beta", "cell_angle_gamma",
        "diffrn_ambient_temp", "diffrn_detector_detector", "diffrn_detector_type", "diffrn_source_source",
        "difradpdbx_scattering_type", "difradpdbx_monochromatic_or_laue_ml", "expcrygrow_pdbx_details",
        "exptl_crystals_number", "rcsentinfo_diffrn_resolution_high_provenance_source",
        "rcsentinfo_diffrn_resolution_high_value", "rcsentinfo_diffrn_radiation_wavelength_maximum",
        "rcsentinfo_diffrn_radiation_wavelength_minimum", "rcsentinfo_nonpolymer_entity_count",
        "rcsentinfo_nonpolymer_molecular_weight_maximum", "rcsentinfo_nonpolymer_molecular_weight_minimum",
        "refhisd_res_high", "refhisd_res_low", "refhisnumber_atoms_solvent", "refhisnumber_atoms_total",
        "refhispdbx_number_atoms_ligand", "refhispdbx_number_atoms_nucleic_acid", "refhispdbx_number_atoms_protein",
        "refine_ls_rfactor_rfree", "refine_ls_rfactor_rwork", "refine_ls_rfactor_obs", "refine_ls_dres_high",
        "refine_ls_dres_low", "reflns_d_resolution_high", "reflns_d_resolution_low", "reflsnumber_reflns_obs",
        "reflspercent_reflns_rfree", "reflspercent_reflns_obs", "reflns_number_obs", "reflns_pdbx_ordinal",
        "symmetry_int_tables_number", "reflns_percent_possible_obs", "rcsentinfo_polymer_molecular_weight_maximum",
        "rcsentinfo_polymer_molecular_weight_minimum", "thicknesserror", "tilterror", "gibbs", "tilt", "thickness",
        "subunit_segments", "refpdbls_sigma_f", "refine_biso_mean", "reflsnumber_reflns_rfree", "refine_overall_suml",
        "refpdbsolvent_shrinkage_radii", "refpdbsolvent_vdw_probe_radii", "reflns_pdbx_redundancy",
        "refpdbnet_iover_sigma_i", "expcrygrow_p_h", "exptl_crystal_grow_temp", "refshed_res_high",
        "refshed_res_low", "reflns_shell_pdbx_ordinal"
    ]
    return [col for col in dynamic_columns_to_drop if col in numerical_data.columns] + additional_columns_to_drop

def drop_columns(data, columns_to_drop):
    return data.drop(columns=columns_to_drop, inplace=False, errors='ignore')

def preprocess_data(data, method="X-ray"):
    # Filter data
    filtered_data = filter_data_by_method(data, method)
    # Clean data
    cleaned_data = clean_data(filtered_data)
    # Separate columns
    numerical_data, categorical_data = separate_columns(cleaned_data)
    # Identify columns to drop
    columns_to_drop = identify_columns_to_drop(numerical_data)
    # Drop columns
    numerical_data = drop_columns(numerical_data, columns_to_drop)
    numerical_data.columns = [clean_column_name(col) for col in numerical_data.columns]
    
    return numerical_data, categorical_data

def feature_boxplot(data, thirty_percent_for_boxplot, categorical_columns=[]):
    # Automatically generate the alias mapping from column names
    feature_aliases = {col: f'{i + 1}' for i, col in enumerate(data.columns) if col != 'x'}
    # Remove keys from dictionary based on values in the list. This is needed for the table
    filtered_dict = {key: value for key, value in feature_aliases.items() if key not in categorical_columns}
    # Transform the dataset for Altair
    df_melted = data.melt(
        id_vars=['group', 'subgroup', 'rcsentinfo_experimental_method', 'pdb_code'], 
        var_name='Feature', value_name='Value'
    )
    # Map the aliases in the DataFrame
    df_melted['Feature'] = df_melted['Feature'].map(feature_aliases)
    # Create a table for the feature mapping
    mapping_df = pd.DataFrame({
        'Feature': list(filtered_dict.keys()),
        'Alias': list(filtered_dict.values())
    })
    # Create the boxplot
    boxplot = alt.Chart(df_melted).mark_boxplot().encode(
        x=alt.X('Feature:O', axis=alt.Axis(format='~s', labelAngle=360)),
        y=alt.Y('Value:Q', axis=alt.Axis(format='~s', labelAngle=360), scale=alt.Scale(type='log')),
        tooltip=categorical_columns
    ).properties(
        width=thirty_percent_for_boxplot,
        title='Boxplot of Selected Features.'
    )
    # Create a text chart for the feature mapping
    mapping_table = alt.Chart(mapping_df).mark_text(align='right').encode(
        y=alt.Y('Alias:N', axis=None),
    ).properties(
        title='Feature Mapping'
    )

    original_feature = mapping_table.encode(text='Feature:N').properties(title=alt.TitleParams(text='Feature', align='right'))
    alias = mapping_table.encode(text='Alias:N').properties(title=alt.TitleParams(text='Alias', align='right'))
    text_data = alt.hconcat(alias, original_feature )
    # Concatenate the boxplot and the mapping table vertically
    combined_chart = alt.vconcat(
        boxplot, text_data
    ).resolve_legend(
        color="independent"
    )
    #.configure_view(strokeWidth=0)

    return combined_chart

def outlier_detection_implementation(
    selected_attributes=[], 
    numerical_data=pd.DataFrame(), 
    categorical_data=pd.DataFrame(), 
    training_attrs=['Component 1', 'Component 2'], 
    plot_attrs=['Component 1', 'Component 2'],
    algorithm="DBSCAN",
    width_chart_single=150,
    width_chart_single2="container",
    create_pairwise_plot_bool=False
):
    column_list = selected_attributes

    df = prepare_data(numerical_data, categorical_data, column_list)
    categorical_columns = [
        "group", "subgroup", 
        "rcsentinfo_experimental_method", 
        "pdb_code"
    ]

    if len(column_list) > 1:
        # pearson_corr = calculate_pearson_correlation(df, column_list)
        # pearson_corr.to_csv("xxxxxxxxx.csv")

        numerical_data_filtered = df[column_list].reset_index(drop=True)
        categorical_data_filtered = df[[
            "group", "subgroup", 
            "rcsentinfo_experimental_method", 
            "pdb_code"
        ]].reset_index(drop=True)
        methods_params = {
            'PCA': {'n_components': 2},
            't-SNE': {'n_components': 2, 'perplexity': 50},
            'UMAP': {'n_components': 2, 'n_neighbors': 15}
        }

        combined_plot_data = perform_dimensionality_reduction(
            numerical_data_filtered, 
            methods_params
        )
        combined_data = merge_reduced_data_with_categorical(
            combined_plot_data, 
            numerical_data_filtered, 
            categorical_data_filtered
        )

        detector = OutlierDetection(training_attrs, plot_attrs)

        if(create_pairwise_plot_bool):
            # PCA is the first followed by TSNE and UMAP
            thirty_percent_for_boxplot = (width_chart_single2 * 0.3)
            seventy_percent_for_boxplot = (width_chart_single2 - thirty_percent_for_boxplot) - 35
            outlier_annotation_tab = detect_outliers_and_data(combined_data[0], algorithm, detector)
            outlier_detection_plot, create_pairwise_plot_chart = create_pairwise_plot(
                outlier_annotation_tab, column_list, categorical_columns, 
                width_chart_single, seventy_percent_for_boxplot,
                plot_attrs, "PCA",
                outlier_chart_title="Outlier detection using " + algorithm
            )
            
            bottom_chart = alt.hconcat(
                outlier_detection_plot,
                feature_boxplot(df, thirty_percent_for_boxplot, categorical_columns)
            )
            
            return alt.vconcat(
                bottom_chart,
                create_pairwise_plot_chart,
            ).resolve_scale(
                color='independent'
            ).configure_view(strokeWidth=0).configure_title(
                font='Arial',
                anchor='middle',
                color='black'
            )
        else:
            # PCA is the first followed by TSNE and UMAP
            # detect_and_visualize_outliers(combined_data[0], 'IsolationForest', detector)
            thirty_percent_for_boxplot = (width_chart_single2 * 0.3)
            seventy_percent_for_boxplot = (width_chart_single2 - thirty_percent_for_boxplot) - 35
            bottom_chart = alt.hconcat(
                detect_and_visualize_outliers(combined_data[0], 'DBSCAN', detector, "PCA", seventy_percent_for_boxplot),
                feature_boxplot(df, thirty_percent_for_boxplot, categorical_columns)
            )
            return bottom_chart.resolve_scale(
                color='independent'
            ).configure_view(strokeWidth=0)
    else:
        return {}
    
    
def protein_regrouping(
    df, extended_column="group", 
    value="TRANSMEMBRANE", conditions = [
        'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL', 
        'TRANSMEMBRANE PROTEINS:BETA-BARREL'
    ]
):  
    # Create a new column and set its value based on the condition
    df['new_' + extended_column] = df[extended_column].apply(
        lambda x: value if x in conditions else x
    )
    
    return df