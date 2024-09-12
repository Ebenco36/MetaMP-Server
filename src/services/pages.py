import pandas as pd
import altair as alt
from vega_datasets import data
from src.services.Helpers.helper import (
    NAPercent,
    countriesD,
    fetch_and_process_chloropleth_data,
    generate_max_upper_for_bin,
    get_lat_long,
    load_and_prepare_data,
    operateD,
    parser_change_dot_to_underscore,
    generate_range_bins,
    generate_list_with_difference,
    convert_to_numeric_or_str,
    convert_to_type,
)

from src.services.graphs.helpers import Graph, convert_chart
from src.Dashboard.data import array_string_type
from src.services.range_order import columns_range_limit
from src.Commands.Migration.classMigrate import Migration
from src.services.Helpers.BasicClasses.GroupByClass import GroupBy
from src.services.data.columns.remove_columns import not_needed_columns


class Pages:

    def __init__(self, data):
        self.data = data
        self.grouping = GroupBy(self.data)
        self.selected_columns_to_vis = []
        self.chunked_data = None


    def transformGroupColumn(self):
        self.data['group'] = self.data['group'].replace({
            'MONOTOPIC MEMBRANE PROTEINS': 1,
            'TRANSMEMBRANE PROTEINS:ALPHA-HELICAL': 2,
            'TRANSMEMBRANE PROTEINS:BETA-BARREL': 3,
        })
        
        return self
    
    def dashboard_helper(self, group_by_column=""):
        # replace dot with underscore
        quantitative_replace_dot_with_underscore = parser_change_dot_to_underscore(
            self.data.columns
        )
        quantitative_data = tuple(quantitative_replace_dot_with_underscore)
        # Group the data by the 'Category' column
        grouped_data = self.data.groupby(group_by_column).size()
        grouped_data = grouped_data.reset_index()
        grouped_data = pd.DataFrame(grouped_data)
        grouped_data.columns = [group_by_column, "Cumulative MP Structures"]

        return grouped_data, group_by_column

    def dashboard_helper_exemption(
        self,
        group_by_column="resolution",
        range_name="range_value",
        range_resolution_meters=0.2,
    ):
        # check if column has * . Separate based on *
        field_filter = ""

        if "*" in group_by_column:
            # Field filter is the value we are looking for within a column
            # This is specific to rcsb_entry_info_selected_polymer_entity_types
            data_split = group_by_column.split("*")
            group_by_column = data_split[0]
            field_filter = data_split[1]
            # exceptional case
            if "rcsentinfo_selected_polymer_entity_types" != group_by_column:
                group_by_column = Migration.shorten_column_name(group_by_column)

        if field_filter != "":
            group_by_column = group_by_column + "*" + field_filter
        if not "*" in group_by_column and group_by_column != "":
            # Apply the custom function to 'Column1'
            self.data[group_by_column] = self.data[group_by_column].apply(
                convert_to_numeric_or_str
            )

            # Convert string list to list
            self.data["rcsentinfo_software_programs_combined"] = self.data[
                "rcsentinfo_software_programs_combined"
            ].apply(lambda x: convert_to_type(x))

            # Separate string column
            mask_str = self.data[group_by_column].apply(lambda x: isinstance(x, str))
            df_numeric = self.data[~mask_str]
            df_str = self.data[mask_str]
            max_value = df_numeric[group_by_column].max(skipna=True)

            if (
                not df_numeric.empty
                and not pd.isna(max_value)
                and group_by_column != "rcsentinfo_software_programs_combined"
                and group_by_column != "exptl_crystal_grow_method"
            ):
                # Convert 'selected column' column to numeric values in the numeric DataFrame
                df_numeric[group_by_column] = pd.to_numeric(
                    df_numeric[group_by_column], errors="coerce"
                )
                max_range_meters = generate_max_upper_for_bin(
                    group_by_column, float(max_value), range_resolution_meters
                )  # add value accordingly so that we can capture all
                range_bins = generate_range_bins(
                    range_resolution_meters, max_range_meters
                )
                generated_list = generate_list_with_difference(
                    len(range_bins), range_resolution_meters
                )
                # Define custom bins for range grouping in the numeric DataFrame
                bins = generated_list
                labels = range_bins
                # Create a new column 'range_name' based on the range of 'Species' values in the numeric DataFrame
                df_numeric[range_name] = pd.cut(
                    df_numeric[group_by_column], bins=bins, labels=labels[:-1]
                )

                # Group by 'group_by_column' in the numeric DataFrame and sum the 'Value' for each range
                grouped_numeric_data = (
                    df_numeric.groupby(range_name, observed=False).size().reset_index()
                )

                grouped_str_data = df_str.groupby(group_by_column).size().reset_index()
                # Concatenate the dataframes vertically
                merged_df = pd.concat(
                    [grouped_numeric_data, grouped_str_data], ignore_index=True
                )

                # Convert 'Column1' to object data type
                merged_df[range_name] = merged_df[range_name].astype("object")
                merged_df[group_by_column] = merged_df[group_by_column].astype("object")

                # Update 'Column1' with 'Column2' values where 'Column1' is NaN
                merged_df[range_name].fillna(merged_df[group_by_column], inplace=True)

                # Drop the 'extra if exist' column
                merged_df.drop(group_by_column, axis=1, inplace=True)
                merged_df.columns = [group_by_column, "Cumulative MP Structures"]
            elif group_by_column in array_string_type():
                # Method 2: Using value_counts
                all_names = [
                    name
                    for names_list in self.data[group_by_column]
                    for name in names_list
                ]
                merged_df = pd.Series(all_names).value_counts().reset_index()
                merged_df.columns = [group_by_column, "Cumulative MP Structures"]
            elif group_by_column == "exptl_crystal_grow_method1":
                data = self.data[
                    (self.data["exptl_crystal_grow_method1"] != "HANGING DROP")
                    & (self.data["exptl_crystal_grow_method1"] != "SITTING DROP")
                ]
                grouped_data = data.groupby("exptl_crystal_grow_method1").size()
                grouped_data = grouped_data.reset_index()
                grouped_data = pd.DataFrame(grouped_data)
                grouped_data.columns = ["exptl_crystal_grow_method1", "Cumulative MP Structures"]
                merged_df = grouped_data
            else:
                # replace dot with underscore
                quantitative_replace_dot_with_underscore = (
                    parser_change_dot_to_underscore(self.data.columns)
                )
                quantitative_data = tuple(quantitative_replace_dot_with_underscore)
                # Group the data by the 'Category' column
                grouped_data = self.data.groupby(group_by_column).size()
                grouped_data = grouped_data.reset_index()
                grouped_data = pd.DataFrame(grouped_data)
                grouped_data.columns = [group_by_column, "Cumulative MP Structures"]
                merged_df = grouped_data
        else:
            # group by column here follows this format parent_tag*search_key
            search_key = group_by_column.split("*")
            merged_df_ = self.data[
                self.data[search_key[0]].str.upper()
                == search_key[1].replace("_", " ").upper()
            ]
            # Group by year and 'Category', and count records
            merged_df = (
                merged_df_.groupby([merged_df_["bibliography_year"], search_key[0]])
                .size()
                .reset_index(name="Cumulative MP Structures")
            )
            group_by_column = "bibliography_year"

            # merged_df = merged_df[['bibliography_year', 'Cumulative MP Structures']]

        # Filter rows where 'Age' column value is greater than zero
        # merged_df = merged_df[merged_df["Cumulative MP Structures"] > 0]

        # # Sort the DataFrame in ascending order based on 'Age'
        merged_df = merged_df.sort_values(by="Cumulative MP Structures")

        # Rearrange the index based on the sorted order
        merged_df = merged_df.reset_index(drop=True)

        return merged_df, group_by_column

    def getMapData(self):
        # Read the data into DataFrame
        df = self.data

        # Assuming 'rcsb_primary_citation_country' is the column containing country names
        grouped_df = df.groupby("rcsb_primary_citation_country")

        # Now you can perform operations on each group, for example, count the number of occurrences of each country
        country_counts = grouped_df.size().reset_index()

        # Rename the column to 'count'
        country_counts = country_counts.rename(columns={0: "count"})

        # Reset the index
        country_counts = country_counts.reset_index(drop=True)

        # Apply the function to create new columns for latitude and longitude
        country_counts[["latitude", "longitude", "location"]] = country_counts[
            "rcsb_primary_citation_country"
        ].apply(lambda x: pd.Series(get_lat_long(x)))

        # Set the index to 'rcsb_primary_citation_country'
        # country_counts = country_counts.set_index('rcsb_primary_citation_country')
        country_counts = country_counts.rename(
            columns={"rcsb_primary_citation_country": "country"}
        )
        # Display the updated DataFrame
        return country_counts

    def releasedStructuresByCountries(self, chart_dict):
        map_df = self.getDataFromChartDict(chart_dict)
        # Create the lollipop chart
        lollipop_chart = (
            alt.Chart(map_df)
            .transform_filter((alt.datum.count > 0) & (alt.datum.country != None))
            .mark_circle(size=50, color="#005EB8")
            .encode(
                x=alt.X(
                    "country:O",
                    axis=alt.Axis(labelAngle=45),
                    sort=alt.EncodingSortField(field="count", order="descending"),
                ),  # Rotate x-axis labels for better readability
                y="count",
                tooltip=alt.Tooltip(
                    "count:Q", title="No. of released Membrane Protein Structure"
                ),
            )
            .properties(
                width="container",
                title=[
                    "Resolved Membrane Protein (MP) Structures",
                    " across different countries.",
                ],
            )
        )

        # Add the vertical lines (lollipops)
        lollipop_lines = (
            alt.Chart(map_df)
            .transform_filter((alt.datum.count > 0) & (alt.datum.country != None))
            .mark_rule(color="#005EB8")
            .encode(
                x=alt.X(
                    "country:O",
                    axis=alt.Axis(labelAngle=45),
                    sort=alt.EncodingSortField(field="count", order="descending"),
                ),  # Rotate x-axis labels for better readability
                y="count",
                tooltip=alt.Tooltip(
                    "count:Q", title="No. of released Membrane Protein Structure"
                ),
            )
        )

        # Combine the chart and lines
        chart = lollipop_chart + lollipop_lines
        # chart.properties(width=1000).save("countryChart.png", scale_factor=2.0)
        # chart to dict for display
        return convert_chart(chart)

    def getMap(self):
        data, map_data = self.viewMap()
        return data, map_data

    def europe_choropleth_map(self):
        country_counts = self.getMapData()
        url = "https://dmws.hkvservices.nl/dataportal/data.asmx/read?database=vega&key=europe"
        # Define data sources from remote origin
        europe = alt.topo_feature(url, "europe")

        country_counts.loc[country_counts["country"] == "UK", "country"] = "GB"
        url = url
        df = fetch_and_process_chloropleth_data(url)
        general = pd.concat([country_counts, df], axis=0)
        # Define basemap

        opacity = alt.condition(
            alt.datum.count
            == 0 | alt.datum.count
            == None,  # Check if value is 0 or None
            alt.value(0.2),
            "count:N",
            legend=None,
        )
        map_base = (
            alt.Chart(europe)
            .mark_geoshape()
            .encode(
                tooltip=[
                    alt.Tooltip("properties.NAME:N", title="Country"),
                    alt.Tooltip("count:Q", title="Value"),
                ],
                # color=alt.Color('count:Q', scale=alt.Scale(scheme='blues', domain=[None, 50]), legend=alt.Legend(title='Values', format='.1%')),
                opacity=opacity,
            )
            .transform_lookup(
                lookup="id", from_=alt.LookupData(general, "country", ["count"])
            )
            .configure_legend(title=None)  # Set the legend title to None to disable it
            .properties(
                width="container",
                title=[
                    "Cumulative Sum of Resolved Membrane Protein (MP) Structures",
                    " Across Different Countries in Europe.",
                ],
            )
            .project("mercator")
        )

        return convert_chart(map_base)

    def viewMap(self):
        all_country_count = pd.merge(
            operateD(self.data), countriesD(), on="iso_code_2", how="outer"
        )
        all_country_count["count"].fillna(0, inplace=True)
        all_country_count["legend"] = (
            all_country_count["count"].astype(str)
            + " ("
            + all_country_count["country"].astype(str)
            + ")"
        )
        all_country_count = all_country_count.dropna(subset=["count"])
        all_country_count["count"] = all_country_count["count"].astype(int)
        # all_country_count.to_csv("countryData.csv", index=False)
        # Load the world map data
        source = alt.topo_feature(data.world_110m.url, "countries")
        base_map = (
            alt.Chart(source)
            .mark_geoshape(stroke="white")
            .transform_lookup(
                lookup="id",
                from_=alt.LookupData(
                    all_country_count,
                    "country_number",
                    [
                        "count",
                        "country",
                        "longitude",
                        "legend_values",
                        "latitude",
                        "iso_code_2",
                        "iso_code_3",
                    ],
                ),
            )
            .encode(
                color=alt.condition(
                    (alt.datum["count"] > 0),
                    alt.Color(
                        "count:Q",
                        scale=alt.Scale(domain=[0, 2000]),
                        legend=alt.Legend(
                            title="Cumulative MP Structures", 
                            tickCount=15, orient="bottom", 
                            direction="horizontal",
                            titlePadding=10,
                            labelPadding=15,
                            gradientLength=300,
                        ),
                    ),  # Include legend configuration
                    alt.value("#D6EAF8"),  # Color for regions with count <= 0
                ),
                tooltip=[
                    alt.Tooltip("country:N", title="Country"),
                    alt.Tooltip("count:Q", title="Count"),
                    alt.Tooltip("iso_code_2:N", title="ISO CODE 2"),
                    alt.Tooltip("iso_code_3:N", title="ISO CODE 3"),
                    alt.Tooltip("longitude:Q", title="Longitude"),
                    alt.Tooltip("latitude:Q", title="Latitude"),
                ],
            )
            .transform_filter(
                alt.datum["count"]
                >= 0 & alt.datum["country"]
                != None  # Filter data for count greater than zero
            )
            .properties(
                width="container",
                title=[
                    "Resolved Membrane Protein (MP) Structures",
                    " across different countries.",
                ],
            )
            .project("mercator")
        )

        annotations = (
            alt.Chart(all_country_count)
            .mark_text(
                fontSize=12,
                fontWeight="bold",
                dx=0,  # Offset for text
                dy=-0,  # Offset for text
            )
            .encode(
                longitude="longitude:Q",
                latitude="latitude:Q",
                text=alt.condition(
                    (alt.datum.count > 0) & (alt.datum.country != None),
                    "flag:N",
                    alt.value(""),  # Empty string if latitude or longitude is null
                ),
                tooltip=[
                    alt.Tooltip("country:N", title="Country"),
                    alt.Tooltip("count:Q", title="Count"),
                    alt.Tooltip("iso_code_2:N", title="ISO CODE 2"),
                    alt.Tooltip("iso_code_3:N", title="ISO CODE 3"),
                    alt.Tooltip("longitude:Q", title="Longitude"),
                    alt.Tooltip("latitude:Q", title="Latitude"),
                ],
            )
        )

        filtered_map = alt.layer(base_map.transform_filter("datum.id !== 10"))

        final_map = (filtered_map + annotations).configure_view(strokeWidth=0)
        # final_map.properties(width=1000).save('globalMapChart.png', scale_factor=2.0)
        map_data = all_country_count[all_country_count["count"] > 0][
            [
                "count",
                "country",
                "country_number",
                "flag",
                "iso_code_2",
                "iso_code_3",
                "latitude",
                "legend",
                "location",
                "longitude",
            ]
        ]
        map_data = map_data.fillna("Unknown Country").sort_values(by="count", ascending=False)
        return map_data.to_dict(orient="records"), convert_chart(final_map)

    
    def view_trends_by_database_year_default(self):
        final_data = load_and_prepare_data()
        
        if "database" in final_data:
            # Rename databases for better readability
            databases = {
                'OPM': 'OPM (Orientation of Proteins in Membranes)',
                'PDB': 'PDB (Protein Data Bank)',
                'UniProt': 'UniProt (Universal Protein Resource)'
            }
            final_data['database'] = final_data['database'].replace(databases)
    
    
        # Base chart for common properties
        base = alt.Chart(final_data).properties(width="container")
        # Line chart for overall trend
        lines = (
            base.mark_line()
            .encode(
                x=alt.X(
                    "bibliography_year:O", title="Year"
                ),  # Ordinal (O) scale for discrete years
                y=alt.Y("count:Q", title="Cumulative MP Structures"),
                color=alt.Color(
                    f"database:N",
                    legend=alt.Legend(
                        title="Database", labelLimit=0, orient="bottom", direction="vertical"
                    ),
                ),
                tooltip=["database", "bibliography_year", "count"],
            )
            .configure_legend(
                orient="bottom", 
                symbolType='square'
            )
            .properties(
                width="container",
                title="Comparative Annual Contributions by PDB, OPM, and UniProt Databases",
            )
        )
        # lines.properties(width=1000).save('annualContribution.png', scale_factor=2.0)
        return convert_chart(lines)
    
    
    def view_trends_by_database_year(self):
        final_data = load_and_prepare_data()
        
        if "database" in final_data:
            # Rename databases for better readability
            databases = {
                'OPM': 'OPM (Orientation of Proteins in Membranes)',
                'PDB': 'PDB (Protein Data Bank)',
                'UniProt': 'UniProt (Universal Protein Resource)'
            }
            final_data['database'] = final_data['database'].replace(databases)
        
        
        # Compute the crosstab with cumulative sum
        d = pd.crosstab(final_data['bibliography_year'], final_data['database'], values=final_data['count'], aggfunc='sum').cumsum()

        # Reset the index to get 'bibliography_year' as a column
        d = d.reset_index()

        # Melt the dataframe to convert it into long format for Altair
        d_melted = d.melt(id_vars='bibliography_year', var_name='database', value_name='count')

        custom_colors = {
            'PDB (Protein Data Bank)': '#005EB8',  # Blue for PDB
            'OPM (Orientation of Proteins in Membranes)': '#A0C4E1',  # Green for OPM
            'UniProt (Universal Protein Resource)': '#F4A261'  # Orange for UniProt
        }
        
        # Create the stacked bar chart using Altair
        chart = alt.Chart(d_melted).mark_bar().encode(
            x=alt.X('bibliography_year:O', title='Year'),
            y=alt.Y('count:Q', title='Cumulative MP Structures'),
            color=alt.Color(
                f"database:N",
                scale=alt.Scale(domain=list(custom_colors.keys()), range=list(custom_colors.values())),
                legend=alt.Legend(
                    title="Database", labelLimit=0, orient="bottom", direction="vertical"
                ),
            ),
            tooltip=[
                alt.Tooltip('database:N', title='Database'),
                alt.Tooltip('bibliography_year:O', title='Year'),
                alt.Tooltip('count:Q', title='Count')
            ]
        ).properties(
            width=800,
            height=400,
            title="Comparative Annual Contributions by PDB, OPM, and UniProt Databases"
        )

        return convert_chart(chart)

    def average_resolution_over_years(self, data):
        data = data.dropna(subset=["processed_resolution", "bibliography_year"])
        if "rcsentinfo_experimental_method" in data:
            # Rename databases for better readability
            experimental_method_list = {
                'EM': 'Cryo-Electron Microscopy (Cryo-EM)',
                'NMR': 'NMR (Nuclear Magnetic Resonance)',
                'X-ray': 'X-ray Crystallography (X-ray)'
            }
            data['rcsentinfo_experimental_method'] = data['rcsentinfo_experimental_method'].replace(experimental_method_list)
            
        data["bibliography_year"] = pd.to_numeric(
            data["bibliography_year"], errors="coerce"
        )
        # Group by year and experimental method, calculate average resolution
        resolution_by_year = (
            data.groupby(["bibliography_year", "rcsentinfo_experimental_method"])[
                "processed_resolution"
            ]
            .agg(["median", "mean", "std", "sum"])
            .reset_index()
        )
        resolution_by_year.columns = [
            "bibliography_year",
            "rcsentinfo_experimental_method",
            "processed_resolution_median",
            "mean",
            "std",
            "Sum",
        ]
        resolution_by_year["processed_resolution_median"] = resolution_by_year[
            "processed_resolution_median"
        ].round(2)
        resolution_by_year["std"] = resolution_by_year["std"].round(2)

        # resolution_by_year.to_csv("mean_resolution_by_year.csv")
        # Plotting
        base = (
            alt.Chart(resolution_by_year)
            .mark_line(point=True, interpolate='monotone')
            .encode(
                x=alt.X("bibliography_year:O", title="Year"),
                y=alt.Y("processed_resolution_median:Q", title="Average Resolution in Angstroms (Å)"),
                color=alt.Color(
                    "rcsentinfo_experimental_method:N",
                    title="Experimental Method",
                    legend=alt.Legend(orient="bottom", direction="horizontal", labelLimit=0)
                ),
                tooltip=[
                    "bibliography_year",
                    "rcsentinfo_experimental_method",
                    "processed_resolution_median",
                ],
            ).properties(
                width="container",
                title="Median resolution improvement by experimental method over time",
            )
        )
        
        # Vertical line annotation for the start of EM data
        start_year = 1996
        annotation = alt.Chart(pd.DataFrame({
            'bibliography_year': [start_year],
            'Value': [3.5]  # Adjust the value to place the annotation at the correct height
        })).mark_rule(color='grey', strokeDash=[5, 5]).encode(
            x='bibliography_year:O'
        )
        
        # Text annotation explaining the start of EM data
        annotation_text = alt.Chart(pd.DataFrame({
            'bibliography_year': [start_year],
            'Value': [3.0],  # Adjust the value to place the text at the correct height
            'Text': ["EM data starts."]
        })).mark_text(align='left', dx=-45, dy=-32, fontWeight='bold', color='darkgrey').encode(
            x='bibliography_year:O',
            y='Value:Q',
            text='Text:N',
        )

        
        chart = alt.layer(base, annotation, annotation_text).configure_legend(
            orient="bottom", 
            labelPadding=15,
            columnPadding=20, 
            padding=15, 
            symbolType='square'
        )
        
        # chart.properties(width=1000).save("averageResolution.png", scale_factor=2.0)
        return convert_chart(chart)

    def view_dashboard(self, get_query_params, conf={}, ranges_={}):
        # Get the URL parameters using Streamlit routing
        selected_content = get_query_params

        range_resolution_meters = (
            columns_range_limit.get(selected_content)
            if columns_range_limit.get(selected_content)
            else 0.2
        )
        df_, pivot_col_ = self.dashboard_helper_exemption(
            selected_content, "range_values", range_resolution_meters
        )

        # Resetting the index to maintain the original order
        df_.reset_index(drop=True, inplace=True)

        # check if Cumulative MP Structures is part of columns
        if "Cumulative MP Structures" in df_.columns:
            sorted_df = df_.sort_values(by="Cumulative MP Structures", ascending=False)[
                int(ranges_.get("from", 0)) : int(ranges_.get("to", 50))
            ]
        else:
            sorted_df = df_.sort_values(by=pivot_col_, ascending=False)[
                int(ranges_.get("from", 0)) : int(ranges_.get("to", 50))
            ]

        chart_obj = Graph.plot_bar_chart(
            sorted_df, pivot_col_, conf, selected_content=selected_content
        )
        # chart_obj.properties(width=1000).save(
        #     selected_content + ".png", scale_factor=2.0
        # )

        return convert_chart(chart_obj), df_

    def remove_emptiness_with_percentage(self, perc=50):
        df = self.data.drop(not_needed_columns, inplace=False, axis=1)
        df = df[df.select_dtypes(include=["float", "int", "float64", "int64"]).columns]
        # get percentage of emptiness
        NA = NAPercent(df)
        NA["NA Percent"]
        NA["NA Percent"] = NA["NA Percent"].astype(float)

        NA.to_csv("NAPercent.csv", index=False)

        df_columns = NA[NA["NA Percent"] < perc].index.tolist()

        # We can use this for further analysis
        df_great = df[df_columns]

        NA.to_csv("NAChunkedDF.csv", index=False)

        return df_great

    def getDataFromChartDict(self, chart_data):
        # Get the first key in the dictionary
        first_key = next(iter(chart_data.get("datasets")))

        # Get the first content of the first key element
        extracted_dataset = chart_data.get("datasets")[first_key]

        # Create a DataFrame from the provided data
        map_df = pd.DataFrame(extracted_dataset)

        return map_df
