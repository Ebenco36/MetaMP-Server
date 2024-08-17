
import random
import numpy as np
import pandas as pd
import altair as alt
from scipy.optimize import curve_fit
from src.services.graphs.helpers import Graph
from src.services.Helpers.BasicClasses.GroupByClass import GroupBy
from src.services.data.columns.quantitative.quantitative import cell_columns, rcsb_entries
from src.services.data.columns.quantitative.quantitative_array import quantitative_array_column
from src.services.Helpers.helper import (
    parser_change_dot_to_underscore,
    generate_color_palette
)

def data_flowxxxx(protein_db, title="Cumulative sum of resolved Membrane Protein (MP) Structures over time"):
    import numpy as np
    d = pd.crosstab(protein_db.bibliography_year, columns=protein_db.group).cumsum()

    d = d.stack().reset_index()
    d = d.rename(columns={0:'CumulativeCount'})
    d = d.convert_dtypes()
    #### Line fit start here
    # Aggregate cumulative counts by year and group
    aggregated_df = d.groupby(['bibliography_year', 'group'])['CumulativeCount'].sum().reset_index()

    # Fit exponential model to the entire combined dataset
    combined_df = aggregated_df.groupby('bibliography_year')['CumulativeCount'].sum().reset_index()

    # Extract year and cumulative count from DataFrame
    years = combined_df['bibliography_year'].values
    cumulative_count = combined_df['CumulativeCount'].values

    # Use data from 1985 to 2004 to fit the exponential growth
    # years_fit = years[:15]  # 1985 to 2004
    # cumulative_count_fit = cumulative_count[:15]
    
    # Find the indices for years between 1985 and 2005 (inclusive)
    
    start_year = 1985
    end_year = 2005
    start_index = np.where(years == start_year)[0][0]
    end_index = np.where(years == end_year)[0][0] + 1  # Including 2005
    # Split the data for fitting
    years_fit = years[start_index:end_index]  # From 1985 to 2005
    cumulative_count_fit = cumulative_count[start_index:end_index]

    # Fit the exponential growth curve
    popt, _ = curve_fit(exp_growth, years_fit, cumulative_count_fit, p0=(100, 0.1))

    # Generate points for the fitted curve
    x_exp = np.arange(min(years), max(years) + 1)
    y_exp = exp_growth(x_exp, *popt)

    # Convert data to DataFrame
    df_fit = pd.DataFrame({
        'Year': x_exp,
        'Fitted Growth': y_exp
    })

    # Determine y-axis limit from the actual data
    y_max = cumulative_count.max()
    df_fit = df_fit[df_fit['Fitted Growth'] <= y_max]

    ##### Line fit ends here+
    
    
    # Define a custom color palette
    start_color = '#005EB8'
    end_color = '#B87200'

    color_list = ['#D9DE84', '#93C4F6', '#005EB8' , '#636B05']
    unique_group_list = list(protein_db['group'].unique())
    # Generate a color palette with 10 colors
    num_colors = len(unique_group_list)
    
    ordered_list = [
        'MONOTOPIC MEMBRANE PROTEINS',
        'TRANSMEMBRANE PROTEINS:BETA-BARREL',
        'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL'
    ]
    
    # Sorting based on custom order
    unique_group_list.sort(key=lambda x: ordered_list.index(x))
    
    palette = generate_color_palette(start_color, end_color, num_colors)
    # random.shuffle(palette)

    custom_palette = alt.Scale(domain=unique_group_list, range=color_list[:num_colors])
    
    entries_over_time = alt.Chart(d).mark_bar().encode(
        x=alt.X('bibliography_year:O', title="Year"),
        y=alt.Y('CumulativeCount:Q', title='Cumulative MP Structures', scale=alt.Scale(domain=[0, y_max])),
        color=alt.Color('group', scale=custom_palette, legend=alt.Legend(title="Groups", labelLimit=0, direction = 'vertical')),
        tooltip=[alt.Tooltip('CumulativeCount:Q'),
                alt.Tooltip('group'),
                alt.Tooltip('bibliography_year:O')]
    )
    
    # Add exponential fit line
    line = alt.Chart(df_fit).mark_line(color='red', strokeDash=[5, 5]).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Fitted Growth:Q', title='Cumulative MP Structures', scale=alt.Scale(domain=[0, y_max])),
        tooltip=['Year', 'Fitted Growth']
    )

    # Combine bar chart and regression line
    chart_with_regression = entries_over_time + line

    
    # Adjust the legend positioning and centering the symbols
    chart_with_regression = chart_with_regression.configure_view(
        stroke='transparent'
    ).configure_legend(
        orient='bottom',
        # padding=20, 
        offset=2,
        titleOrient='top',
        labelLimit=0,
        # labelFontSize=12,
        # titleFontSize=14,
        # symbolSize=100,
        # #labelAlign='center',  # Center align the labels
        # labelBaseline="middle",
        # titleAlign="center",
        # titleAnchor="middle",
        # symbolOffset=10  # Adjust this value to better center the symbols
    ).properties(
        width="container",
        title=title
    )
    # chart_with_regression.properties(width=1000).save('cumulativeChart.png', scale_factor=2.0)
    return chart_with_regression.to_dict(format="vega")

def data_flow(protein_db, title="Cumulative sum of resolved Membrane Protein (MP) Structures over time"):
    import numpy as np
    d = pd.crosstab(protein_db.bibliography_year, columns=protein_db.group).cumsum()
    ordered_list = [
        'MONOTOPIC MEMBRANE PROTEINS',
        'TRANSMEMBRANE PROTEINS:BETA-BARREL',
        'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL'
    ]
    d = d.stack().reset_index()
    d = d.rename(columns={0:'CumulativeCount'})
    d = d.convert_dtypes()
    d = extend_dataframe_with_missing_years(d, ordered_list, view_type="partly")
    d['index'] = pd.factorize(d['bibliography_year'])[0]
    # Aggregate cumulative counts by year and group
    aggregated_df = d.groupby(['bibliography_year', 'group', 'index'])['CumulativeCount'].sum().reset_index()

    # Fit exponential model to the entire combined dataset
    combined_df = aggregated_df.groupby(['bibliography_year', 'index'])['CumulativeCount'].sum().reset_index()

    
    # Fit exponential model
    indices = combined_df['index'].values
    cumulative_count = combined_df['CumulativeCount'].values
    
    start_year, end_year = 0, 20
    start_index = np.where(indices == start_year)[0][0]
    end_index = np.where(indices == end_year)[0][0] + 1
    years_fit = indices[start_index:end_index]
    cumulative_count_fit = cumulative_count[start_index:end_index]
    popt, _ = curve_fit(exp_growth, years_fit, cumulative_count_fit, p0=(100, 0.1))
    x_exp = np.arange(min(indices), max(indices) + 1)
    y_exp = exp_growth(x_exp, *popt)
    
    df_fit = pd.DataFrame({'Index': x_exp, 'Fitted Growth': y_exp})
    y_max = cumulative_count.max()
    df_fit = df_fit[df_fit['Fitted Growth'] <= y_max]

    color_list = ['#D9DE84', '#93C4F6', '#005EB8' , '#636B05']
    unique_group_list = list(protein_db['group'].unique())
    # Generate a color palette with 10 colors
    num_colors = len(unique_group_list)
    unique_group_list.sort(key=lambda x: ordered_list.index(x))
    custom_palette = alt.Scale(domain=unique_group_list, range=color_list[:num_colors])
    
    entries_over_time = alt.Chart(d).mark_bar(size=10).encode(
        x=alt.X('index:Q', title="Year since first structure (1985)", axis=alt.Axis(labelAngle=0, tickCount=6),
            scale=alt.Scale(domain=[0, d['index'].max()])
        ),
        y=alt.Y('CumulativeCount:Q', title='Cumulative MP Structures', 
                scale=alt.Scale(domain=[0, y_max])
        ),
        color=alt.Color('group:N', scale=custom_palette, legend=alt.Legend(title="Group", orient="bottom", direction="vertical")),
        tooltip=[alt.Tooltip('bibliography_year:O'), alt.Tooltip('CumulativeCount:Q'), alt.Tooltip('index:O')]
    ).interactive()
    
    # Add exponential fit line
    line = alt.Chart(df_fit).mark_line(color='red', strokeDash=[5, 5]).encode(
        x=alt.X('Index:Q', title="Year since first structure (1985)", axis=alt.Axis(labelAngle=0, tickCount=6),
            scale=alt.Scale(domain=[0, d['index'].max()])
        ),
        y=alt.Y('Fitted Growth:Q', title='Cumulative MP Structures', 
                scale=alt.Scale(domain=[0, y_max])
            ),
        tooltip=['Index', 'Fitted Growth']
    )

    # Combine bar chart and regression line
    chart_with_regression = entries_over_time + line

    # Adjust the legend positioning and centering the symbols
    chart_with_regression = chart_with_regression.configure_view(
        stroke='transparent'
    ).configure_legend(
        orient='bottom',
        # padding=20, 
        offset=2,
        titleOrient='top',
        labelLimit=0
    ).properties(
        width="container",
        title=title
    )
    # chart_with_regression.properties(width=1000).save('cumulativeChart.png', scale_factor=2.0)
    return chart_with_regression.to_dict(format="vega")

def group_data_by_methods(df, columns=['bibliography_year', 'rcsentinfo_experimental_method'], col_color="rcsentinfo_experimental_method", col_x="bibliography_year", chart_type="line", bin_value=None, interactive=False, arange_legend="vertical"):
    # Group and count the data
    group_subtype_count = df.groupby(columns).size().reset_index(name='Cumulative MP Structures')

    if "rcsentinfo_experimental_method" in group_subtype_count:
        # Rename methods for better readability
        method_renames = {
            'EM': 'Cryo-Electron Microscopy (Cryo-EM)',
            'Multiple methods': 'Multi-Method',
            'NMR': 'Nuclear Magnetic Resonance (NMR)',
            'X-ray': 'X-ray Crystallography (X-ray)'
        }
        group_subtype_count['rcsentinfo_experimental_method'] = group_subtype_count['rcsentinfo_experimental_method'].replace(method_renames)
    
    # Define chart title based on the presence of 'rcsentinfo_experimental_method'
    title = 'Experimental Method' if "rcsentinfo_experimental_method" in group_subtype_count else "Classes"

    # Choose chart type
    chart_t = alt.Chart(group_subtype_count).mark_line() if chart_type == "line" else alt.Chart(group_subtype_count).mark_bar()

    # Generate tick positions for x-axis
    tick_positions = list(range(int(df['bibliography_year'].min()), int(df['bibliography_year'].max()) + 1, 3))

    # Configure x-axis
    bin = alt.Bin(maxbins=int(bin_value)) if bin_value else None
    axis = alt.Axis(labelAngle=-45, tickCount=15, values=tick_positions if not bin_value else None)

    # Define custom colors
    custom_colors = {
        'Cryo-Electron Microscopy (Cryo-EM)': '#517caa',
        'Multi-Method': '#e55e5d',
        'Nuclear Magnetic Resonance (NMR)': '#73b7b3',
        'X-ray Crystallography (X-ray)': '#f38820'       
    }
        
    # Create the chart object
    chart_obj = chart_t.encode(
        x=alt.X(f'{col_x}:O', title='Bibliography (Year)', sort="x", bin=bin, axis=axis),
        y=alt.Y('Cumulative MP Structures:Q', title="Cumulative MP Structures"),
        tooltip=['Cumulative MP Structures:Q'],
        color=alt.Color(f'{col_color}:N', legend=alt.Legend(title=title, labelLimit=0, direction=arange_legend),
            scale=alt.Scale(domain=list(custom_colors.keys()), range=list(custom_colors.values()))
        )
    )

    # Add interactivity if required
    chart_interactive = chart_obj.interactive() if interactive else chart_obj
        
    chart = chart_interactive.configure_legend(
        orient='bottom',
        symbolType='square'
        # symbolSize = 200,
        # symbolStrokeWidth=5
    ).properties(
        width="container",
        title=["Cumulative MP(s) resolved via various experimental methods over time,", " with each line representing a distinct method"]
    )
    
    # chart.properties(width=1000).save('by_methods.png', scale_factor=2.0)
        
    return chart.to_dict()

# Define exponential function for fitting
def exp_growthxxxx(x, a, b):
    return a * np.exp(b * (x - 2005))

def create_combined_chart_cumulative_growthxxxx(protein_db, chart_width=800):
    # Ensure 'group' column exists and update it
    if 'group' not in protein_db.columns:
        raise KeyError("The 'group' column is missing from the DataFrame.")
    
    protein_db['group'] = protein_db['group'].replace({
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
    protein_db['label'] = protein_db['group'].map(group_labels)
    
    
    # Define a custom color palette
    color_list = ['#D9DE84', '#93C4F6', '#005EB8']
    unique_group_list = list(protein_db['label'].unique())
    num_colors = len(unique_group_list)
    ordered_list = [
        'Group 1 (MONOTOPIC MEMBRANE PROTEINS)',
        'Group 2 (TRANSMEMBRANE PROTEINS:BETA-BARREL)',
        'Group 3 (TRANSMEMBRANE PROTEINS:ALPHA-HELICAL)'
    ]
    unique_group_list.sort(key=lambda x: ordered_list.index(x))
    custom_palette = alt.Scale(domain=unique_group_list, range=color_list[:num_colors])
    
    
    unique_group_list_group = list(protein_db['group'].unique())
    num_colors_group = len(unique_group_list_group)
    ordered_list_group = [
        'Group 1',
        'Group 2',
        'Group 3'
    ]
    unique_group_list_group.sort(key=lambda x: ordered_list_group.index(x))
    custom_palette_group = alt.Scale(domain=unique_group_list_group, range=color_list[:num_colors_group])
    
    
    # Check if 'label' column was successfully created
    if 'label' not in protein_db.columns:
        raise KeyError("The 'label' column was not created successfully.")
    
    # Group by 'label' and count occurrences
    grouped_data = protein_db.groupby(["label", "group"]).size().reset_index(name='CumulativeCount')
    
    # Group by 'taxonomic domain' and count occurrences
    grouped_data_taxonomic_domain = protein_db[["taxonomic_domain", "label", "group"]].groupby(["taxonomic_domain", "label", "group"]).size().reset_index(name="CumulativeCount")
    
    # Sort by count
    grouped_data = grouped_data.sort_values(by='CumulativeCount', ascending=True)


    #############################3rd Chart###########################
    #################################################################
    
    # Prepare the data for the cumulative chart with exponential fit
    d = pd.crosstab([protein_db["bibliography_year"]], columns=[protein_db["group"], protein_db["label"]]).cumsum()

    # Reset the index to convert 'bibliography_year' back to a column
    d = d.reset_index()

    # Melt the crosstab to get 'group' and 'label' as individual columns
    d = d.melt(
        id_vars=["bibliography_year"], 
        var_name=["group", "label"], 
        value_name="CumulativeCount"
    )
    
    aggregated_df = d.groupby(['bibliography_year', 'label'])['CumulativeCount'].sum().reset_index()
    combined_df = aggregated_df.groupby('bibliography_year')['CumulativeCount'].sum().reset_index()
    
    # Fit exponential model
    years = combined_df['bibliography_year'].values
    cumulative_count = combined_df['CumulativeCount'].values
    start_year, end_year = 1985, 2005
    start_index = np.where(years == start_year)[0][0]
    end_index = np.where(years == end_year)[0][0] + 1
    years_fit = years[start_index:end_index]
    cumulative_count_fit = cumulative_count[start_index:end_index]
    popt, _ = curve_fit(exp_growth, years_fit, cumulative_count_fit, p0=(100, 0.1))
    x_exp = np.arange(min(years), max(years) + 1)
    y_exp = exp_growth(x_exp, *popt)
    df_fit = pd.DataFrame({'Year': x_exp, 'Fitted Growth': y_exp})
    y_max = cumulative_count.max()
    df_fit = df_fit[df_fit['Fitted Growth'] <= y_max]
    
    # Create a brush selection
    brush = alt.selection_interval(encodings=['x', 'y'])
    unique_group_list = list(protein_db['label'].unique())
    
    # Calculate the available width by subtracting the padding
    padding = 200 
    available_width = chart_width - padding

    chart_width_1 = 0.5*available_width
    chart_width_2 = 0.5*chart_width_1
    
    grouped_bar_chart = alt.Chart(grouped_data).mark_bar().encode(
        x=alt.X(
            'group:N', title='Group', sort=None, axis=alt.Axis(
                labelAngle=0,
                labelLimit=0
            )
        ),
        y=alt.Y('CumulativeCount:Q', title='Cumulative MP Structures', scale=alt.Scale(domain=[0, y_max])),
        color=alt.Color('label:N', scale=custom_palette, legend=alt.Legend(
            title="Group", 
            orient="bottom", 
            labelLimit=0, 
            direction="vertical"
            )
        ),
        tooltip=["label", "CumulativeCount"]
    ).add_params(
        brush
    ).properties(
        title=['Cumulative resolved MPs', ' categorized by group'],
        width=chart_width_2
    )
    
    grouped_bar_chart_taxonomic = alt.Chart(grouped_data_taxonomic_domain).mark_bar().encode(
        x=alt.X(
            'taxonomic_domain:N', title='Taxonomic Domain', sort=None, axis=alt.Axis(
                labelAngle=15,
                labelLimit=0
            ), 
        ),
        y=alt.Y('CumulativeCount:Q', title=None, scale=alt.Scale(domain=[0, y_max])),
        color=alt.Color('label:N', scale=custom_palette, legend=None),
        tooltip=["label", "taxonomic_domain", "CumulativeCount"]
    ).transform_filter(
        brush
    ).properties(
        title=['Cumulative resolved MPs ', 'categorized by taxonomic domain'],
        width=chart_width_2
    ).interactive()
    
    
    entries_over_time = alt.Chart(d).mark_bar().encode(
        x=alt.X('bibliography_year:O', title="Year"),
        y=alt.Y('CumulativeCount:Q', title=None, scale=alt.Scale(domain=[0, y_max])),
        color=alt.Color('label:N', scale=custom_palette, legend=None),
        tooltip=[alt.Tooltip('CumulativeCount:Q'), alt.Tooltip('label'), alt.Tooltip('bibliography_year:O')]
    ).transform_filter(
        brush
    ).properties(
        width=chart_width_1,
        title=["Cumulative resolved Membrane ", "Protein (MP) Structures over time"]
    )
    
    # Add exponential fit line
    line = alt.Chart(df_fit).mark_line(color='red', strokeDash=[5, 5]).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Fitted Growth:Q', title='Cumulative MP Structures', scale=alt.Scale(domain=[0, y_max])),
        tooltip=['Year', 'Fitted Growth']
    )
    # Combine the bar chart and the exponential fit line
    chart_with_regression = entries_over_time + line
    
    # Combine both charts into one visualization
    combined_chart = alt.hconcat(
        grouped_bar_chart, 
        grouped_bar_chart_taxonomic,  
        chart_with_regression
    ).configure_view(
        stroke='transparent'
    ).configure_legend(
        orient='bottom',
        offset=2,
        titleOrient='top',
        labelLimit=0
    )

    return combined_chart.to_dict()

def exp_growth(x, a, b):
    return a * np.exp(b * (x - 20))

def extend_dataframe_with_missing_years(df, groups, view_type="all"):
    """
    Extend the dataframe to include missing years with specified groups and labels.

    Parameters:
    df (pd.DataFrame): The original dataframe with at least 'bibliography_year' column.
    groups (dict): A dictionary where keys are group identifiers and values are group labels.

    Returns:
    pd.DataFrame: The extended dataframe including missing years and group information.
    """
    # Identify existing years in the dataframe
    existing_years = df['bibliography_year'].unique()
    min_year, max_year = min(existing_years), max(existing_years)
    all_years = set(range(min_year, max_year + 1))
    missing_years = all_years - set(existing_years)
    
    # Create new entries for missing years
    new_entries = []
    if view_type == "all":
        for year in missing_years:
            for group, label in groups.items():
                new_entries.append({
                    'bibliography_year': year,
                    'group': group,
                    'label': label,
                    'CumulativeCount': 0
                })
    else:
        for year in missing_years:
            for group in groups:
                new_entries.append({
                    'bibliography_year': year,
                    'group': group,
                    'CumulativeCount': 0
                })
    
    # Convert new entries to DataFrame
    missing_df = pd.DataFrame(new_entries)
    
    # Combine the original DataFrame with the new entries
    df_extended = pd.concat([df, missing_df], ignore_index=True)
    
    # Sort by 'bibliography_year' if needed
    df_extended = df_extended.sort_values(by='bibliography_year').reset_index(drop=True)
    
    return df_extended

def create_combined_chart_cumulative_growth(protein_db, chart_width=1000):
    # Ensure 'group' column exists and update it
    if 'group' not in protein_db.columns:
        raise KeyError("The 'group' column is missing from the DataFrame.")
    
    protein_db['group'] = protein_db['group'].replace({
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
    protein_db['label'] = protein_db['group'].map(group_labels)
    
    # Define a custom color palette
    color_list = ['#D9DE84', '#93C4F6', '#005EB8']
    unique_group_list = list(protein_db['label'].unique())
    num_colors = len(unique_group_list)
    ordered_list = [
        'Group 1 (MONOTOPIC MEMBRANE PROTEINS)',
        'Group 2 (TRANSMEMBRANE PROTEINS:BETA-BARREL)',
        'Group 3 (TRANSMEMBRANE PROTEINS:ALPHA-HELICAL)'
    ]
    unique_group_list.sort(key=lambda x: ordered_list.index(x))
    custom_palette = alt.Scale(domain=unique_group_list, range=color_list[:num_colors])
    
    # Check if 'label' column was successfully created
    if 'label' not in protein_db.columns:
        raise KeyError("The 'label' column was not created successfully.")
    
    # Group by 'label' and count occurrences
    grouped_data = protein_db.groupby(["label", "group"]).size().reset_index(name='CumulativeCount')
    
    # Group by 'taxonomic domain' and count occurrences
    grouped_data_taxonomic_domain = protein_db[["taxonomic_domain", "label", "group"]].groupby(["taxonomic_domain", "label", "group"]).size().reset_index(name="CumulativeCount")
    
    # Prepare the data for the cumulative chart with exponential fit
    
    # Extend dataframe
    d = pd.crosstab([protein_db["bibliography_year"]], columns=[protein_db["group"], protein_db["label"]]).cumsum()
    d = d.reset_index()
    d = d.melt(
        id_vars=["bibliography_year"], 
        var_name=["group", "label"], 
        value_name="CumulativeCount"
    )
    d = extend_dataframe_with_missing_years(d, group_labels)
    # Convert bibliography_year to an index
    d['index'] = pd.factorize(d['bibliography_year'])[0]
    
    aggregated_df = d.groupby(['index', 'label', 'bibliography_year'])['CumulativeCount'].sum().reset_index()
    combined_df = aggregated_df.groupby(['index', 'bibliography_year'])['CumulativeCount'].sum().reset_index()
    
    # Fit exponential model
    indices = combined_df['index'].values
    cumulative_count = combined_df['CumulativeCount'].values
    
    start_year, end_year = 0, 20
    start_index = np.where(indices == start_year)[0][0]
    end_index = np.where(indices == end_year)[0][0] + 1
    years_fit = indices[start_index:end_index]
    cumulative_count_fit = cumulative_count[start_index:end_index]
    popt, _ = curve_fit(exp_growth, years_fit, cumulative_count_fit, p0=(100, 0.1))
    x_exp = np.arange(min(indices), max(indices) + 1)
    y_exp = exp_growth(x_exp, *popt)
    
    df_fit = pd.DataFrame({'Index': x_exp, 'Fitted Growth': y_exp})
    y_max = cumulative_count.max()
    df_fit = df_fit[df_fit['Fitted Growth'] <= y_max]
    
    # Create a brush selection
    brush = alt.selection_interval(encodings=['x', 'y'])
    unique_group_list = list(protein_db['label'].unique())
    
    # Calculate the available width by subtracting the padding
    padding = 200 
    available_width = chart_width - padding

    chart_width_1 = 0.5*available_width
    chart_width_2 = 0.5*chart_width_1
    
    grouped_bar_chart = alt.Chart(grouped_data).mark_bar().encode(
        x=alt.X(
            'group:N', title='Group', sort=None, axis=alt.Axis(
                labelAngle=0,
                labelLimit=0
            )
        ),
        y=alt.Y('CumulativeCount:Q', title='Cumulative MP Structures', scale=alt.Scale(domain=[0, y_max])),
        color=alt.Color('label:N', scale=custom_palette, legend=alt.Legend(
            title="Group", 
            orient="bottom", 
            labelLimit=0, 
            direction="vertical"
            )
        ),
        tooltip=["label", "CumulativeCount"]
    ).add_params(
        brush
    ).properties(
        title=['Cumulative resolved MPs', ' categorized by group'],
        width=chart_width_2
    )
    
    grouped_bar_chart_taxonomic = alt.Chart(grouped_data_taxonomic_domain).mark_bar().encode(
        x=alt.X(
            'taxonomic_domain:N', title='Taxonomic Domain', sort=None, axis=alt.Axis(
                labelAngle=15,
                labelLimit=0
            ), 
        ),
        y=alt.Y('CumulativeCount:Q', title=None, scale=alt.Scale(domain=[0, y_max])),
        color=alt.Color('label:N', scale=custom_palette, legend=None),
        tooltip=["label", "taxonomic_domain", "CumulativeCount"]
    ).transform_filter(
        brush
    ).properties(
        title=['Cumulative resolved MPs ', 'categorized by taxonomic domain'],
        width=chart_width_2
    ).interactive()
    
    entries_over_time = alt.Chart(d).mark_bar(size=10).encode(
        x=alt.X('index:Q', title="Year since first structure (1985)", axis=alt.Axis(labelAngle=0, tickCount=6),
            scale=alt.Scale(domain=[0, d['index'].max()])
        ),
        y=alt.Y('CumulativeCount:Q', title=None, 
                scale=alt.Scale(domain=[0, y_max])
        ),
        color=alt.Color('label:N', scale=custom_palette, legend=None),
        tooltip=[alt.Tooltip('bibliography_year:O'), alt.Tooltip('CumulativeCount:Q'), alt.Tooltip('label'), alt.Tooltip('index:O')]
    ).transform_filter(
        brush
    ).properties(
        width=chart_width_1,
        title=["Cumulative resolved Membrane ", "Protein (MP) Structures over time"]
    ).interactive()
    
    # Add exponential fit line
    line = alt.Chart(df_fit).mark_line(color='red', strokeDash=[5, 5]).encode(
        x=alt.X('Index:Q', title="Year since first structure (1985)", axis=alt.Axis(labelAngle=0, tickCount=6),
            scale=alt.Scale(domain=[0, d['index'].max()])
        ),
        y=alt.Y('Fitted Growth:Q', title='Cumulative MP Structures', 
                scale=alt.Scale(domain=[0, y_max])
            ),
        tooltip=['Index', 'Fitted Growth']
    )
    # Combine the bar chart and the exponential fit line
    chart_with_regression = entries_over_time + line
    
    # Combine both charts into one visualization
    combined_chart = alt.hconcat(
        grouped_bar_chart, 
        grouped_bar_chart_taxonomic,  
        chart_with_regression
    ).configure_view(
        stroke='transparent'
    ).configure_legend(
        orient='bottom',
        offset=2,
        titleOrient='top',
        labelLimit=0
    ).configure_concat(
        spacing=2  # Remove spacing between charts
    )

    return combined_chart.to_dict()