# resources.py

import uuid
import altair
import pandas as pd
import altair as alt
from scipy import stats
from flask import g, request
from datetime import timedelta
from src.Dashboard.services import (
    get_items, 
    get_table_as_dataframe, 
    get_tables_as_dataframe
)
from src.Training.services import (
    OptionService,
    CategoryService,
    QuestionService,
    preprocess_data,
    group_annotation,
    FilterToolService, 
    protein_regrouping,
    transform_dataframe,
    getChartForQuestion,
    UserResponseService,
    create_visualization,  
    get_records_on_method,
    get_quantification_data,
    filter_questions_in_sets, 
    aggregate_inconsistencies,
    outlier_detection_implementation
)
from utils.redisCache import RedisCache
from src.MP.services import DataService
from src.Training.models import Category
from src.utils.response import ApiResponse
from flask_restful import Resource, reqparse
from src.services.graphs.helpers import Graph, convert_chart
from src.Training.serializers import (
    FilterToolSchema, QuestionSchema, CategorySchema
)
from src.middlewares.auth_middleware import token_required
from src.services.basic_plots import group_data_by_methods
from src.MP.machine_learning_services import UnsupervisedPipeline, plotCharts


question_parser = reqparse.RequestParser()
question_parser.add_argument('text', type=str, help='Text of the question')
answer_parser = reqparse.RequestParser()
answer_parser.add_argument('text', type=str, help='Text of the answer')
answer_parser.add_argument('question_id', type=int, help='question id')
answer_parser.add_argument('is_correct', type=bool, help='Is the answer correct')

filter_tool_parser = reqparse.RequestParser()
filter_tool_parser.add_argument('title', type=str, required=True, help='Title is required')
filter_tool_parser.add_argument('name', type=str, required=True, help='Name is required')
filter_tool_parser.add_argument('selected_option', type=str, required=True, help='Selected option is required')
filter_tool_parser.add_argument('question_id', type=int, required=True, help='Question ID is required')

class CategoryResource(Resource):
    @token_required
    def get(self, category_id):
        category = CategoryService.get_category(category_id)
        category_schema = CategorySchema()
        return category_schema.dump(category)
    
    @token_required
    def put(self, category_id):
        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str, required=True, help='Name cannot be blank')
        parser.add_argument('description', type=str, required=False)
        args = parser.parse_args()

        # Check if the category with the given ID exists
        existing_category = CategoryService.get_category(category_id)
        if not existing_category:
            return {"message": "Category not found"}, 404

        # Update the category
        updated_category = CategoryService.update_category(category_id, args['name'], args['description'])

        if updated_category:
            category_schema = CategorySchema()  # Replace with your actual CategorySchema
            return category_schema.dump(updated_category)
        else:
            return {"message": "Cannot update category. Another category with the same name may exist."}, 400
        
    @token_required
    def delete(self, category_id):
        deleted = CategoryService.delete_category(category_id)
        if deleted:
            return {"message": f"Category deleted successfully."}
        else:
            return {"message": "Cannot delete category. Questions are attached."}, 400

class CategoryListResource(Resource):
    # @token_required
    def get(self):
        question_type = request.args.get('type', 'training')
        categories = CategoryService.get_all_categories()

        # Filter categories based on the specified question_type
        """
        filtered_categories = [category for category in categories if any(
            question.question_type == question_type for question in category.questions
        )]
        """

        category_schema = CategorySchema(many=True)
        return category_schema.dump(categories)
        return filter_questions_in_sets(question_sets=category_schema.dump(categories), class_name=question_type)

    @token_required
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str, required=True, help='Name cannot be blank')
        parser.add_argument('description', type=str, required=True, help='Description cannot be blank')
        args = parser.parse_args()

        new_category = CategoryService.create_category(args['name'], args['description'])
        if new_category:
            category_schema = CategorySchema()
            return category_schema.dump(new_category), 201
        else:
            return {"message": "Category with this name already exists."}, 400

class QuestionResource(Resource):
    @token_required
    def get(self, question_id):
        question = QuestionService.get_question_by_id(question_id)
        question_schema = QuestionSchema()
        return question_schema.dump(question), 200
    
    @token_required
    def put(self, question_id):
        question_parser.add_argument('text', type=str, required=True, help='Question text')
        question_parser.add_argument('instructions', type=str, required=False, help='Question Instructions')
        question_parser.add_argument('hints', type=str, required=False, help='Question Hints')
        question_parser.add_argument('item_order', type=int, required=True, help='Question order')
        question_parser.add_argument('question_type', type=str, required=False, help='Question Type')
        question_parser.add_argument('category_id', type=str, required=True, help='Question category')
        args = question_parser.parse_args()
        question = QuestionService.get_question_by_id(question_id)
        if question:
            category = Category.query.get_or_404(args['category_id'])
            QuestionService.update_question(
                question, args['text'], 
                args['item_order'], 
                category, args['question_type'], 
                args['instructions'],
                args['hints']
            )
            return {'message': 'Question updated successfully'}
        return {'message': 'Question not found'}, 404

    @token_required
    def delete(self, question_id):
        question = QuestionService.get_question_by_id(question_id)
        if question:
            QuestionService.delete_question(question)
            return {'message': 'Question deleted successfully'}
        return {'message': 'Question not found'}, 404

class QuestionsResource(Resource):
    @token_required
    def get(self):
        questions = QuestionService.get_all_questions()
        question_schema = QuestionSchema(many=True)
        return question_schema.dump(questions), 200

    @token_required
    def post(self):
        question_parser.add_argument('text', type=str, required=True, help='Question text')
        question_parser.add_argument('category_id', type=str, required=True, help='Question category')
        question_parser.add_argument('item_order', type=int, required=True, help='Question order')
        question_parser.add_argument('instructions', type=str, required=False, help='Question Instructions')
        question_parser.add_argument('hints', type=str, required=False, help='Question Hints')
        question_parser.add_argument('question_type', type=str, required=False, help='Question Type')
        args = question_parser.parse_args()
        category = Category.query.get_or_404(args['category_id'])
        question = QuestionService.create_question(
            args['text'], 
            category,
            args['item_order'], 
            args['question_type'], 
            args['instructions'],
            args['hints']
        )
        return {
            'id': question.id, 
            'text': question.text, 
            'category_id': question.category_id,
            'item_order': question.item_order, 
            'question_type': question.question_type, 
            'instructions': question.instruction,
            'hints': question.hints,
            'created_at': question.created_at.isoformat(),
            'created_at': question.updated_at.isoformat(),
            'options': question.options
        }, 201

class OptionResource(Resource):
    
    @token_required
    def get(self, answer_id):
        option = OptionService.get_option_by_id(answer_id)
        if(option):
            return {
                    'id': option.id, 
                    'text': option.text, 
                    'is_correct': option.is_correct,
                    'question_id': option.question_id,
                    'created_at': option.created_at.isoformat(),
                    'created_at': option.updated_at.isoformat(),
            }, 200
        else: 
            return {'message': 'Option not found'}, 404

    @token_required
    def put(self, answer_id):
        args = answer_parser.parse_args()
        answer = OptionService.get_option_by_id(answer_id)
        if answer:
            OptionService.update_option(answer, args['text'], args['is_correct'])
            return {'message': 'Option updated successfully'}
        return {'message': 'Option not found'}, 404

    @token_required
    def delete(self, answer_id):
        answer = OptionService.get_option_by_id(answer_id)
        if answer:
            OptionService.delete_option(answer)
            return {'message': 'Option deleted successfully'}
        return {'message': 'Option not found'}, 404

class OptionsResource(Resource):
    @token_required
    def get(self):
        options = OptionService.get_all_options()
        return [{'id': a.id, 'text': a.text, 'is_correct': a.is_correct, 'question_id': a.question_id} for a in options]

    @token_required
    def post(self):
        args = answer_parser.parse_args()
        option = OptionService.create_option(args['text'], args['question_id'], args['is_correct'])
        return {'id': option.id, 'text': option.text, 'is_correct': option.is_correct, 'question_id': option.question_id}, 201

# FilterToolView
class FilterToolResource(Resource):
    def post(self):
        args = filter_tool_parser.parse_args()

        # Check if the question with the given ID exists
        question = QuestionService.get_question_by_id(args['question_id'])
        if not question:
            return {"message": "Question not found"}, 404

        # Create a new filter tool
        filter_tool = FilterToolService.create_filter_tool(
            args['title'], args['name'], args['parent'], args['selected_option'], args['question_id'], args['options']
        )

        filter_tool_schema = FilterToolSchema()
        return filter_tool_schema.dump(filter_tool), 201

    def get(self, filter_tool_id):
        # Get filter tool with options
        filter_tool = FilterToolService.get_filter_tool(filter_tool_id)

        filter_tool_schema = FilterToolSchema()
        return filter_tool_schema.dump(filter_tool), 200

    def put(self, filter_tool_id):
        args = filter_tool_parser.parse_args()

        # Update filter tool
        FilterToolService.update_filter_tool(
            filter_tool_id, args['title'], args['name'], args['parent'], args['selected_option'], args['options']
        )

        filter_tool = FilterToolService.get_filter_tool(filter_tool_id)
        filter_tool_schema = FilterToolSchema()
        return filter_tool_schema.dump(filter_tool), 200

    def delete(self, filter_tool_id):
        # Delete filter tool
        FilterToolService.delete_filter_tool(filter_tool_id)
        return {"message": "Filter Tool deleted successfully"}, 200

    
    
class UserAnswerResource(Resource):
    @token_required
    def post(self):
        data = request.get_json()
        session_id = request.headers.get('X-Session-ID') or str(uuid.uuid4())
        current_user = g.current_user
        response = UserResponseService.check_user_response(current_user.id, session_id, data)
        return response
    
class UserResponsesResource(Resource):
    @token_required
    def get(self, user_id):
        user_responses = UserResponseService.get_user_responses(user_id)
        return [{'question_id': ur.question_id, 'answer_id': ur.answer_id} for ur in user_responses]


class TestingResources(Resource):
    def get(self):
        data = get_records_on_method()
        return data
    
    
def chartForTraining(
    chart_data, x="resolution", y="rcsentinfo_molecular_weight", 
    tooltips=["rcsentinfo_molecular_weight", 'resolution', 'group', 'pdb_code' ], color="group"
):
    chart = altair.Chart(chart_data).mark_point().encode(
        x=altair.Y(x), 
        y=altair.Y(y),
        color=color,
        tooltip=[altair.Tooltip(tooltip, title=tooltip.capitalize()) for tooltip in tooltips]
    ).properties(
        width="container"
    ).interactive().configure_legend(
        orient='bottom'
    )
    return convert_chart(chart)

class generateChartForQuestions(Resource):
    def __init__(self):
        self.cache = RedisCache()
        
    def post(self):
        data = request.get_json()
        group_by = ""
        if data["type"] and data["type"] == "training":
            
            if data["question"] == 1:
                # Creating a bar chart using Altair
                variable = data["variables"].get("groupby")
                
                ###########################################
                # Apply Cache correctly to first question #
                ###########################################
                
                """
                    Cache Keys Management
                """
                cache_key = "question1_key_" + variable
                # Set expiration time for cache if used
                ttl_in_seconds = timedelta(days=10).total_seconds()
                
                question1_cached_result = self.cache.get_item(cache_key)
                
                if question1_cached_result:
                    variable_counts = pd.DataFrame(question1_cached_result)
                else:
                    dataset = getChartForQuestion(column=variable, filter="")
                    view_data = [
                        'resolution', 'bibliography_year', 
                        'group', 'rcsentinfo_molecular_weight', 
                        "pdb_code", "refine_ls_rfactor_rfree", 
                        "rcsentinfo_experimental_method", "taxonomic_domain"
                    ]
                    
                    chart_data = dataset[view_data]
                    # Grouping data by 'exptl_method' and counting occurrences
                    variable_counts = chart_data[variable].value_counts().reset_index()
                    variable_counts.columns = [variable, 'Cumulative MP Structures']

                    # Store the result in the cache
                    self.cache.set_item(cache_key, variable_counts.to_dict(), ttl=ttl_in_seconds)  # Cache for 10 days
                
                if("rcsentinfo_experimental_method" in variable_counts):
                    variable_counts['rcsentinfo_experimental_method'] = variable_counts['rcsentinfo_experimental_method'].replace({
                        'EM': 'Cryo-Electron Microscopy (Cryo-EM)',
                        'X-ray': 'X-Ray Crystallography',
                        'NMR': 'Nuclear Magnetic Resonance (NMR)',
                        'Multiple methods': 'Multi-methods',
                    })        
                chart = altair.Chart(variable_counts).mark_bar().encode(
                    x=altair.X(
                        variable + ':N', 
                        axis=altair.Axis(
                            labelAngle=30, 
                            title=variable.replace("rcsentinfo", "").replace("_", " ").title(),
                            labelLimit=0
                        )
                    ),
                    y=altair.Y(
                        'Cumulative MP Structures:Q', 
                        axis=altair.Axis(
                            title='Cumulative MP Structures'
                        )
                    ),
                    color=alt.Color(variable + ':N', legend=alt.Legend(title=variable.replace("rcsentinfo", " ").replace("_", " ").title())),
                    tooltip=[variable + ':N', 'Cumulative MP Structures:Q']
                ).properties(
                    width="container"
                ).interactive().configure_legend(
                    orient='bottom', 
                    direction = 'vertical', 
                    labelLimit=0
                )
            # Outlier training
            elif data["question"] == 3:
                variable = data["variable"]
                
                ###########################################
                # Apply Cache correctly to second question#
                ###########################################
                
                """
                    Cache Keys Management
                """
                cache_key = "question3_key_" + variable
                # Set expiration time for cache if used
                ttl_in_seconds = timedelta(days=10).total_seconds()
                
                question3_cached_result = self.cache.get_item(cache_key)
                
                if question3_cached_result:
                    x_ray_data = pd.DataFrame(question3_cached_result)
                else:
                    # Select data for cryo-electron microscopy (EM)
                    x_ray_data = getChartForQuestion(column="rcsentinfo_experimental_method", filter=variable)

                    # Convert 'Resolution' column to numeric, and filter out non-numeric values
                    x_ray_data.loc[:, 'resolution'] = pd.to_numeric(x_ray_data['resolution'], errors='coerce')
                    x_ray_data = x_ray_data.dropna(subset=['resolution'])

                    # Handle non-positive values before applying logarithmic scale
                    x_ray_data['resolution'] = x_ray_data['resolution'].apply(lambda x: max(x, 1e-10))

                    # Calculate group-specific median of 'Resolution'
                    group_median_resolution = x_ray_data.groupby('group')['resolution'].median().reset_index()
                    group_median_resolution.columns = ['group', 'Group_Median_Resolution']

                    # Merge the median values back into the main dataframe
                    x_ray_data = pd.merge(x_ray_data, group_median_resolution, on='group')

                    # Calculate group-specific z-scores
                    x_ray_data['Resolution_Z'] = x_ray_data.groupby('group')['resolution'].transform(lambda x: stats.zscore(x))

                    # Identify and filter potential outliers based on a z-score threshold and group-specific median resolution
                    outliers = x_ray_data[(abs(x_ray_data['Resolution_Z']) > x_ray_data['Group_Median_Resolution']) & (x_ray_data['resolution'] > x_ray_data['Group_Median_Resolution'])]
                    # Store the result in the cache
                    self.cache.set_item(cache_key, x_ray_data.to_dict(), ttl=ttl_in_seconds)  # Cache for 10 days
                    
                # Altair boxplot with logarithmic scale
                x_ray_data['group'] = x_ray_data['group'].replace({
                    'MONOTOPIC MEMBRANE PROTEINS': 1,
                    'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL': 2,
                    'TRANSMEMBRANE PROTEINS:BETA-BARREL': 3,
                })
                chart_obj = altair.Chart(x_ray_data).mark_boxplot().encode(
                    x=altair.X('group:N', title="Group", axis=alt.Axis(labelAngle=360, labelFontSize=9)),
                    y=altair.Y('resolution:Q', title="Resolution (Angstrom (Å))", scale=altair.Scale(type='log')),
                    color=alt.value("#005EB8"),
                    tooltip=['group:N', 'resolution:Q', 'pdb_code:N']
                ).properties(
                    width="container",
                    title='Boxplot of Resolution for ' + variable + ' within Each Group (Log Scale)'
                )
                
                chart = group_annotation(chart_obj)
            
            elif(data["question"] == 8):
                groups = data["variables"].get('groups', None)
                sub_group = data["variables"].get('sub_group', None)
                taxonomic_domain = data["variables"].get('taxonomic_domain', None)
                data = {
                    "search_terms": {
                        "group": groups,
                        "subgroup": sub_group,
                        "taxonomic_domain": taxonomic_domain
                    },
                    "page": data["page"]
                }
                records = get_items(data)
                return records
            
            elif(data["question"] == "NOTNEEDED"):
                chart_width = data.get("chart_width", 800)
                variable = data["variables"].get("features", 
                [
                    'emt_molecular_weight', 
                    'reconstruction_num_particles',
                    'processed_resolution'
                ])
                variable = ['molecular_weight' if var == 'emt_molecular_weight' else var for var in variable]
                
                all_data, _, _, _, _ = DataService.get_data_from_DB()
                numerical_data, categorical_data = preprocess_data(all_data, "EM")
                
                width_chart_single = (chart_width / len(variable)) - 70
                return convert_chart(outlier_detection_implementation(
                    variable, numerical_data, 
                    categorical_data, 
                    training_attrs=['Component 1', 'Component 2'], 
                    plot_attrs=['Component 1', 'Component 2'],
                    width_chart_single=width_chart_single,
                    width_chart_single2=(chart_width - 70),
                    create_pairwise_plot_bool=False
                ))
            
            elif(data["question"] == 5):
                chart_width = data.get("chart_width", 800)
                variable = data["variables"].get("features", [
                    'emt_molecular_weight', 
                    'reconstruction_num_particles',
                    'processed_resolution'
                ])
                variable = ['molecular_weight' if var == 'emt_molecular_weight' else var for var in variable]
                all_data, _, _, _, _ = DataService.get_data_from_DB()
                numerical_data, categorical_data = preprocess_data(all_data, "EM")
                width_chart_single = (chart_width / len(variable)) - 70
                return convert_chart(outlier_detection_implementation(
                    variable, numerical_data, 
                    categorical_data, 
                    training_attrs=['Component 1', 'Component 2'], 
                    plot_attrs=['Component 1', 'Component 2'],
                    width_chart_single=width_chart_single,
                    width_chart_single2=(chart_width - 70),
                    create_pairwise_plot_bool=True
                ))
                
            elif(data["question"] == "DI1"):
                chart_width = data.get("chart_width", 800)
                variable = data["variables"].get("methods")
                all_data, _, _, _, _ = DataService.get_data_from_DB()
                # Define a dictionary to map keywords in famsupclasstype_type_name to expected group values
                expected_groups = {
                    'Monotopic': 'MONOTOPIC MEMBRANE PROTEINS',
                    'Transmembrane': 'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL',
                    'Transmembrane': 'TRANSMEMBRANE PROTEINS:BETA-BARREL'
                    # Add more mappings as needed
                }

                # Assume all_data is already defined and loaded with appropriate data
                dtd = all_data[all_data["rcsentinfo_experimental_method"] == variable]
                df_combined = dtd[[
                    "pdb_code", "famsupclasstype_type_name", 
                    "family_superfamily_classtype_name", 
                    "group", "bibliography_year",
                    "rcsentinfo_experimental_method"
                ]].copy()
                df_combined.dropna(inplace=True)

                # Aggregate inconsistencies
                inconsistencies_by_year = aggregate_inconsistencies(df_combined, expected_groups)

                # Transform the aggregated data
                transformed_data = transform_dataframe(inconsistencies_by_year)

                # Create and display the visualization
                chart_with_table = create_visualization(transformed_data, chart_width)
                
                return convert_chart(chart_with_table)
            
            elif(data["question"] == "DM33"):
                data_type = "X-ray"
                n_components = data["variables"].get('n_components', 2)
                n_neighbors = data["variables"].get('n_neighbors', 20)
                perplexity = data["variables"].get('perplexity', 30)
                action = data["variables"].get('DRT', "tsne_algorithm")
                group_by = data["variables"].get('group_by', "group")
                
                """
                    Cache Keys Management
                """
                extra_field = str(perplexity) if action == "tsne_algorithm" else str(n_neighbors)
                cache_key = "datasets-dimensionalityReduction-" + action + data_type + extra_field + '__' + str(group_by) + "-csv"
                # Set expiration time for cache if used
                ttl_in_seconds = timedelta(days=10).total_seconds()
                
                dm_cached_result = self.cache.get_item(cache_key)
                
                get_column_tag = action.upper().split("_")[0] if action != "tsne_algorithm" else "t-SNE"
                dr_columns = [ get_column_tag + " " + str(char) for char in range(1, 3)]
            
                if dm_cached_result:
                    data = dm_cached_result
                else:
                    result_df_mpstruc_pdb = get_table_as_dataframe("membrane_proteins")
                    result_df_opm = get_table_as_dataframe("membrane_protein_opm")
                    result_df = pd.merge(right=result_df_mpstruc_pdb, left=result_df_opm.drop(columns=["resolution", "name"]), on="pdb_code")
                    result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")
                    all_data = pd.merge(right=result_df, left=result_df_uniprot, on="pdb_code")
                    data_frame = all_data #[all_data["rcsentinfo_experimental_method"] == data_type]
                    data_frame = data_frame[
                        (data_frame["group"] == "TRANSMEMBRANE PROTEINS:BETA-BARREL") |
                        (data_frame["group"] == "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL")
                    ]
                    
                    # data_frame = protein_regrouping(data_frame)
                    
                    result = (
                        UnsupervisedPipeline(data_frame)
                        .fix_missing_data()
                        .variable_separation()
                        .feature_selection(target=group_by)
                        .dimensionality_reduction(
                            reduction_method=action,
                            DR_n_components = int(n_components),
                            perplexity = float(perplexity),
                            n_neighbors=int(n_neighbors), dr_columns=dr_columns,
                            data_type=data_type
                        )
                        #.cluster_data(method='agglomerative')
                        .dm_data()
                    )
                    data = result.data_combined.to_dict(orient="records")
                    if(action == "tsne_algorithm"):
                        # Store the result in the cache
                        self.cache.set_item(cache_key, data, ttl=ttl_in_seconds)  # Cache for 10 days
                    elif(action == "umap_algorithm"):
                        # Store the result in the cache
                        self.cache.set_item(cache_key, data, ttl=ttl_in_seconds)  # Cache for 10 days
                chart_obj = pd.DataFrame(data)
                
                var = "t-SNE " if (action == "tsne_algorithm") else "UMAP " if (action == "umap_algorithm") else "PCA "
                
                chart = plotCharts(chart_obj, class_group=group_by, variables=[var + str(1), var + str(2)])
                    
                return chart
        
        elif data["type"] and data["type"] == "test":
            
            if data["question"] and data["question"] == 2:
                # Creating a bar chart using Altair
                variable = data["variables"].get("groupby") if (data["variables"] and data["question"] == 2) else "rcsentinfo_experimental_method"
                # variable must be categorical
                mark_type = data["variables"].get("chartType") if (data["variables"] and data["question"] == 3) else "line"
                
                bin_value = data["variables"].get("bin_value") if (data["variables"] and data["question"] == 3) else None

                dataset = getChartForQuestion(column=variable, filter="")
                view_data = [
                    'resolution', 'bibliography_year', 
                    'group', 'rcsentinfo_molecular_weight', 
                    "pdb_code", "refine_ls_rfactor_rfree", 
                    "rcsentinfo_experimental_method", "taxonomic_domain"
                ]
                
                chart_data = dataset[view_data]

                # Grouping data by 'rcsb_entry_info_experimental_method' and 'Group' and counting occurrences
                group_method_year_counts = chart_data.groupby([variable, 'bibliography_year']).size().reset_index(name='count')

                # Choose mark type dynamically
                if mark_type == "line":
                    chart = altair.Chart(group_method_year_counts).mark_line()
                else:
                    chart = altair.Chart(group_method_year_counts).mark_bar()
                
                if("rcsentinfo_experimental_method" in group_method_year_counts):
                    group_method_year_counts['rcsentinfo_experimental_method'] = group_method_year_counts['rcsentinfo_experimental_method'].replace({
                        'EM': 'Cryo-Electron Microscopy (Cryo-EM)',
                        'X-ray': 'X-Ray Crystallography',
                        'NMR': 'Nuclear Magnetic Resonance (NMR)',
                        'Multiple methods': 'Multi-methods',
                    }) 
                    group_method_year_counts = group_method_year_counts.rename(columns={
                        'rcsentinfo_experimental_method': 'Experimental Method',
                    })
                    
                chart = alt.Chart.from_dict(
                    group_data_by_methods(
                        chart_data, 
                        columns = [
                            'bibliography_year', 
                            variable
                        ], 
                        col_color=variable, 
                        chart_type=mark_type, 
                        bin_value=bin_value, 
                        interactive=True,
                        arange_legend="vertical"
                    )
                )
            
            elif data["question"] == 4:
                variable = data["variables"].get("methods")
                
                ###########################################
                # Apply Cache correctly to second question#
                ###########################################
                
                """
                    Cache Keys Management
                """
                cache_key = "question4_key_" + variable
                # Set expiration time for cache if used
                ttl_in_seconds = timedelta(days=10).total_seconds()
                
                question4_cached_result = self.cache.get_item(cache_key)
                
                if question4_cached_result:
                    cryo_em_data = pd.DataFrame(question4_cached_result)
                else:
                    # Select data for cryo-electron microscopy (EM)
                    cryo_em_data = getChartForQuestion(column="rcsentinfo_experimental_method", filter=variable)

                    # Convert 'Resolution' column to numeric, and filter out non-numeric values
                    cryo_em_data.loc[:, 'resolution'] = pd.to_numeric(cryo_em_data['resolution'], errors='coerce')
                    cryo_em_data = cryo_em_data.dropna(subset=['resolution'])

                    # Handle non-positive values before applying logarithmic scale
                    cryo_em_data['resolution'] = cryo_em_data['resolution'].apply(lambda x: max(x, 1e-10))

                    # Calculate group-specific median of 'Resolution'
                    group_median_resolution = cryo_em_data.groupby('group')['resolution'].median().reset_index()
                    group_median_resolution.columns = ['group', 'Group_Median_Resolution']

                    # Merge the median values back into the main dataframe
                    cryo_em_data = pd.merge(cryo_em_data, group_median_resolution, on='group')

                    # Calculate group-specific z-scores
                    cryo_em_data['Resolution_Z'] = cryo_em_data.groupby('group')['resolution'].transform(lambda x: stats.zscore(x))

                    # Identify and filter potential outliers based on a z-score threshold and group-specific median resolution
                    outliers = cryo_em_data[(abs(cryo_em_data['Resolution_Z']) > cryo_em_data['Group_Median_Resolution']) & (cryo_em_data['resolution'] > cryo_em_data['Group_Median_Resolution'])]

                    # Store the result in the cache
                    self.cache.set_item(cache_key, cryo_em_data.to_dict(), ttl=ttl_in_seconds)  # Cache for 10 days
                
                cryo_em_data['group'] = cryo_em_data['group'].replace({
                    'MONOTOPIC MEMBRANE PROTEINS': 1,
                    'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL': 2,
                    'TRANSMEMBRANE PROTEINS:BETA-BARREL': 3,
                })
                  
                # Altair boxplot with logarithmic scale
                chart_obj = altair.Chart(cryo_em_data).mark_boxplot().encode(
                    x=altair.X('group:N', title="Group", axis=alt.Axis(labelAngle=360, labelFontSize=9)),
                    y=altair.Y('resolution:Q', title="Resolution (Angstrom (Å))", scale=altair.Scale(type='log')),
                    color=alt.value("#005EB8"),
                    tooltip=['group:N', 'resolution:Q', 'pdb_code:N']
                ).properties(
                    width="container",
                    title='Boxplot of Resolution for ' + variable + ' within Each Group (Log Scale)'
                )
                chart = group_annotation(chart_obj)
            
            elif(data["question"] == 9):
                groups = data["variables"].get('groups', None)
                sub_group = data["variables"].get('sub_group', None)
                taxonomic_domain = data["variables"].get('taxonomic_domain', None)
                data = {
                    "search_terms": {
                        "group": groups,
                        "subgroup": sub_group,
                        "taxonomic_domain": taxonomic_domain
                    }
                }
                records = get_items(data)
                return records
            
            elif(data["question"] == 6):
                chart_width = data.get("chart_width", 800)
                variable = data["variables"].get("features", [
                    # "cell_length_a", 
                    # "cell_length_b", 
                    # "cell_length_c", 
                    "crystal_density_matthews",
                    "molecular_weight", 
                    "processed_resolution"
                ])
                all_data, _, _, _, _ = DataService.get_data_from_DB()
                numerical_data, categorical_data = preprocess_data(all_data, "X-ray")
                width_chart_single = (chart_width / len(variable)) - 70
                return convert_chart(outlier_detection_implementation(
                    variable, numerical_data, 
                    categorical_data, 
                    training_attrs=['Component 1', 'Component 2'], 
                    plot_attrs=['Component 1', 'Component 2'],
                    width_chart_single=width_chart_single,
                    width_chart_single2=(chart_width - 70),
                    create_pairwise_plot_bool=True
                ))
            
            elif(data["question"] == "DI2"):
                chart_width = data.get("chart_width", 800)
                variable = data["variables"].get("methods")
                all_data, _, _, _, _ = DataService.get_data_from_DB()
                # Define a dictionary to map keywords in famsupclasstype_type_name to expected group values
                expected_groups = {
                    'Monotopic': 'MONOTOPIC MEMBRANE PROTEINS',
                    'Transmembrane': 'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL',
                    'Transmembrane': 'TRANSMEMBRANE PROTEINS:BETA-BARREL'
                    # Add more mappings as needed
                }

                # Assume all_data is already defined and loaded with appropriate data
                dtd = all_data[all_data["rcsentinfo_experimental_method"] == variable]
                df_combined = dtd[[
                    "pdb_code", "famsupclasstype_type_name", 
                    "family_superfamily_classtype_name", 
                    "group", "bibliography_year", 
                    "rcsentinfo_experimental_method"
                ]].copy()
                df_combined.dropna(inplace=True)

                # Aggregate inconsistencies
                inconsistencies_by_year = aggregate_inconsistencies(df_combined, expected_groups)

                # Transform the aggregated data
                transformed_data = transform_dataframe(inconsistencies_by_year)

                # Create and display the visualization
                chart_with_table = create_visualization(transformed_data, chart_width)
                
                return convert_chart(chart_with_table)
                
            elif(data["question"] == "DM1"):
                data_type = "EM"
                n_components = data["variables"].get('n_components', 2)
                n_neighbors = data["variables"].get('n_neighbors', 20)
                perplexity = data["variables"].get('perplexity', 50)
                action = data["variables"].get('DRT', "tsne_algorithm")
                group_by = data["variables"].get('group_by', "group")
                """
                    Cache Keys Management
                """
                extra_field = str(perplexity) if action == "tsne_algorithm" else str(n_neighbors)
                cache_key = "datasets-dimensionalityReduction-" + action + data_type + extra_field + '__' + str(group_by) + "-csv"
                # Set expiration time for cache if used
                ttl_in_seconds = timedelta(days=10).total_seconds()
                
                dm_cached_result = self.cache.get_item(cache_key)
                
                get_column_tag = action.upper().split("_")[0] if action != "tsne_algorithm" else "t-SNE"
                dr_columns = [ get_column_tag + " " + str(char) for char in range(1, 3)]
            
                if dm_cached_result:
                    data = dm_cached_result
                else:
                    result_df_mpstruc_pdb = get_table_as_dataframe("membrane_proteins")
                    result_df_opm = get_table_as_dataframe("membrane_protein_opm")
                    result_df = pd.merge(right=result_df_mpstruc_pdb, left=result_df_opm.drop(columns=["resolution", "name"]), on="pdb_code")
                    result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")
                    all_data = pd.merge(right=result_df, left=result_df_uniprot, on="pdb_code")
                    data_frame = all_data[all_data["rcsentinfo_experimental_method"] == data_type]
                    data_frame = data_frame[
                        (data_frame["group"] == "TRANSMEMBRANE PROTEINS:BETA-BARREL") |
                        (data_frame["group"] == "TRANSMEMBRANE PROTEINS:ALPHA-HELICAL")
                    ]
                    
                    result = (
                        UnsupervisedPipeline(data_frame)
                        .fix_missing_data(method_type=data_type)
                        .variable_separation()
                        .feature_selection(target=group_by)
                        .onehot_encoding()
                        .dimensionality_reduction(
                            reduction_method=action,
                            DR_n_components = int(n_components),
                            perplexity = float(perplexity),
                            n_neighbors=int(n_neighbors), dr_columns=dr_columns,
                            data_type=data_type
                        )
                        #.cluster_data(method='agglomerative')
                        .dm_data()
                    )
                    
                    data = result.data_combined.to_dict(orient="records")
                    if(action == "tsne_algorithm"):
                        # Store the result in the cache
                        self.cache.set_item(cache_key, data, ttl=ttl_in_seconds)  # Cache for 10 days
                    elif(action == "umap_algorithm"):
                        # Store the result in the cache
                        self.cache.set_item(cache_key, data, ttl=ttl_in_seconds)  # Cache for 10 days
                chart_obj = pd.DataFrame(data)
                chart = plotCharts(chart_obj, class_group=group_by)
                    
                return chart
            
            elif(data["question"] == "DM3"):
                n_components = data["variables"].get('n_components', 2)
                n_neighbors = data["variables"].get('n_neighbors', 20)
                perplexity = data["variables"].get('perplexity', 50)
                action = data["variables"].get('DRT', "tsne_algorithm")
                group_by = data["variables"].get('group_by', "group")
                """
                    Cache Keys Management
                """
                extra_field = str(perplexity) if action == "tsne_algorithm" else str(n_neighbors)
                cache_key = "datasets-dimensionalityReduction-" + action + data_type + extra_field + '__' + str(group_by) + "-csv"
                # Set expiration time for cache if used
                ttl_in_seconds = timedelta(days=10).total_seconds()
                
                dm_cached_result = self.cache.get_item(cache_key)
                
                get_column_tag = action.upper().split("_")[0] if action != "tsne_algorithm" else "t-SNE"
                dr_columns = [ get_column_tag + " " + str(char) for char in range(1, 3)]
            
                if dm_cached_result:
                    data = dm_cached_result
                else:
                    result_df_mpstruc_pdb = get_table_as_dataframe("membrane_proteins")
                    result_df_opm = get_table_as_dataframe("membrane_protein_opm")
                    result_df = pd.merge(right=result_df_mpstruc_pdb, left=result_df_opm.drop(columns=["resolution", "name"]), on="pdb_code")
                    result_df_uniprot = get_table_as_dataframe("membrane_protein_uniprot")
                    all_data = pd.merge(right=result_df, left=result_df_uniprot, on="pdb_code")
                    data_frame = all_data
                    
                    result = (
                        UnsupervisedPipeline(data_frame)
                        .fix_missing_data()
                        .variable_separation()
                        .feature_selection(target=group_by)
                        .dimensionality_reduction(
                            reduction_method=action,
                            DR_n_components = int(n_components),
                            perplexity = float(perplexity),
                            n_neighbors=int(n_neighbors), dr_columns=dr_columns,
                            data_type=data_type
                        )
                        #.cluster_data(method='agglomerative')
                        .dm_data()
                    )
                    
                    data = result.data_combined.to_dict(orient="records")
                    if(action == "tsne_algorithm"):
                        # Store the result in the cache
                        self.cache.set_item(cache_key, data, ttl=ttl_in_seconds)  # Cache for 10 days
                    elif(action == "umap_algorithm"):
                        # Store the result in the cache
                        self.cache.set_item(cache_key, data, ttl=ttl_in_seconds)  # Cache for 10 days
                chart_obj = pd.DataFrame(data)
                chart = plotCharts(chart_obj, class_group=group_by)
                    
                return chart

        cache_key = "test-" + data["type"] + "-and-question-" + str(data["question"]) + str(variable) + '__' + str(group_by)
        ttl_in_seconds = timedelta(days=10).total_seconds()
        cached_result = self.cache.get_item(cache_key)
        if cached_result:
            return cached_result
        else:
            self.cache.set_item(cache_key, convert_chart(chart), ttl=ttl_in_seconds)  # Cache for 10 days
        
        try:
            response = convert_chart(chart)
        except ValueError as e:
            response = convert_chart(chart)
        return response
            
            