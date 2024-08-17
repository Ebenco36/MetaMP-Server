import altair as alt
import json
from src.services.Helpers.helper import remove_underscore_change_toupper, \
    format_string_caps
from src.services.exceptions.AxisExceptions import AxisException
from src.services.exceptions.NotFoundOnList import NotFoundOnList
from src.services.exceptions.TagDoesnotExist import TagDoesnotExist

class Graph:

    def __init__(self, data = None, axis:list = [], labels:str = "", selection_avenue_default:list = [], selection_type_default:list = []):
        self.data = data
        self.x = None
        self.y = None
        self.z = None # for multi dimension
        self.w = None # for multi dimension
        self.altair_obj = None
        self.selection_avenue = "click"
        self.encoded_x = None
        self.encoded_y = None

        self.axis   = axis
        self.labels = labels
        # defaults
        self.selection_avenue_default   = selection_avenue_default
        self.selection_type_default     = selection_type_default

        # acceptable tags
        self.acceptable_encoding_tags = ["norminal", "temporal", "quantitative", "ordinal"]
        self.start_color = '#005EB8'  # Start Color
        self.end_color = '#B87200'    # End Color 
        self.selection = None


    def set_properties(self, axis:list = [], labels:str = "", selection_avenue_default:list = [], selection_type_default:list = []):
        self.axis   = axis
        self.labels = labels
        # defaults
        self.selection_avenue_default   = selection_avenue_default
        self.selection_type_default     = selection_type_default
        
        return self

    def encoding_tags(self, encoding_tags:list=[], tooltips:list = []):
        if (len(self.axis) == len(encoding_tags) or len(encoding_tags) == 0):
            """
                This services can still be optimized.
            """
            # Title label
            title_x = remove_underscore_change_toupper(tooltips[0])
            title_y = remove_underscore_change_toupper(tooltips[1])

            # check with default tags if exist
            if(len(encoding_tags) > 0):
                x_tag, y_tag = encoding_tags
                my_x_axis = self.x+":"+str(x_tag[0]).capitalize()
                my_y_axis = self.y+":"+str(y_tag[0]).capitalize()
                
                if x_tag in self.acceptable_encoding_tags and y_tag in self.acceptable_encoding_tags:
                    self.encoded_x = alt.X(my_x_axis, title = title_x)
                    self.encoded_y = alt.Y(my_y_axis, title = title_y)
                else:
                    raise TagDoesnotExist("This tag with the name "+x_tag+" or "+y_tag+" does not exist. Acceptable tags includes: "+", ".join(self.acceptable_encoding_tags))
            else:
                my_x_axis = self.x
                my_y_axis = self.y
                self.encoded_x = alt.X(my_x_axis, title = title_x)
                self.encoded_y = alt.Y(my_y_axis, title = title_y)
        else:
            raise AxisException("Axis specified is not equal to the selected encoding tags.")


    def scatter_plot(self):
        if (len(self.axis) != 2):
            raise AxisException("Axis specified is not equal to the selected dimension.")
        
        self.x, self.y = self.axis
        self.altair_obj = alt.Chart(self.data).mark_point()
        return self
    
    def line_plot(self):
        self.x, self.y = self.axis
        self.altair_obj = alt.Chart(self.data).mark_line()
        return self
    
    def area_plot(self):
        self.x, self.y = self.axis
        self.altair_obj = alt.Chart(self.data).mark_area()
        return self
    
    def bar_plot(self):
        self.x, self.y = self.axis
        self.altair_obj = alt.Chart(self.data).mark_bar()
        return self
    
    def circle_plot(self):
        self.x, self.y = self.axis
        self.altair_obj = alt.Chart(self.data).mark_circle()
        return self
    
    def rect_plot(self):
        self.x, self.y = self.axis
        self.altair_obj = alt.Chart(self.data).mark_rect()
        return self
    
    def box_plot(self):
        self.x, self.y = self.axis
        self.altair_obj = alt.Chart(self.data).mark_boxplot()
        return self
    
    def pie_splot(self):
        self.x, self.y = self.axis
        self.altair_obj = alt.Chart(self.data).mark_pie()
        return self
    
    
    def violin_plot(self):
        self.x, self.y = self.axis
        self.altair_obj = alt.Chart(self.data).transform_density(
            self.x,
            as_=[self.x, self.y],
            extent=[5, 50],
            groupby=['Origin']
        ).mark_area(orient='horizontal')
        return self
    
    def point_plot(self):
        self.x, self.y = self.axis
        self.altair_obj = alt.Chart(self.data).mark_point()
        return self

    
    def encoding(self, tooltips:list = [], encoding_tags:list = []):

        # manage color patterns
        # selection = alt.selection_point(fields=[self.y])
        # color = alt.condition(
        #     selection,
        #     alt.Y(self.y).legend(None),
        #     alt.value('lightgray')
        # )
        color = alt.condition(self.selection, self.labels+':N', alt.value('lightgray'), legend=alt.Legend(title=self.labels, columns=5, columnPadding=20, labelLimit=0, direction = 'vertical'))

        # set tooltips

        tooltip_list = [alt.Tooltip(tooltip, title=format_string_caps(tooltip.capitalize())) for tooltip in tooltips]

        # setting encoding tags

        self.encoding_tags(encoding_tags, tooltips)
        if (len(self.axis) == 2):
            self.altair_obj = self.altair_obj.encode(
                self.encoded_x,
                self.encoded_y,
                color=color,
                tooltip=tooltip_list,
                # shape=self.encoded_x
            )
        # This does not exist right now.
        elif(len(self.axis) == 3):
            """Possible we might encounter cases like this 3D"""
            self.altair_obj = self.altair_obj.encode(
                alt.X(self.x),
                alt.Y(self.y),
                alt.Z(self.z),
                color=color,
                tooltip=tooltip_list,
                # shape=self.encoded_x
            )

        elif(len(self.axis) == 4):
            """Possible we might encounter cases like this 4D"""
            self.altair_obj = self.altair_obj.encode(
                alt.X(self.x),
                alt.Y(self.y),
                alt.Z(self.z),
                alt.W(self.w),
                color=color,
                tooltip=tooltip_list,
                # shape=self.encoded_x
            )
        else:
            self.altair_obj = self.altair_obj.encode(
                alt.X(self.x),
                alt.Y(self.y),
                color=color,
                tooltip=tooltip_list,
                # shape=self.encoded_x
            )

        return self
    
    def get_selection_avenue(self, selection_avenue = "drag"):
        if(selection_avenue not in self.selection_avenue_default):
            raise NotFoundOnList("selected option is not on the list")
        self.selection_avenue = selection_avenue
        return self
    
    def set_selection(self, type:str = 'single', groups:list=[]):
        """
        if(type not in self.selection_type_default):
            raise NotFoundOnList("selected option is not on the list")
        """
        if(type == 'single'):
            self.selection = alt.selection_point(on=self.selection_avenue, name='MySelection', fields=groups)
        elif(type == 'multiple'):
            self.selection = alt.selection_multi(on=self.selection_avenue, name='MySelection', fields=groups)
        elif(type == 'interval'):
            self.selection = alt.selection_interval(on=self.selection_avenue, name='MySelection', fields=groups)
        else:
            self.selection = alt.selection_point(on=self.selection_avenue, name='MySelection', fields=groups)
        
        return self
            
    def add_selection(self):
        
        self.altair_obj = self.altair_obj.add_params(self.selection)

        return self
    
    def properties(self, width=200, title = ""):
        # Set the width and height of the chart
        self.altair_obj = self.altair_obj.properties(
            title=title,
            width=width if width > 0 else "container",  # Set the width
            # height=height # set the height of the graph
        )
        return self


    def config(self, label_font_size=12, title_font_size=14, font_size=16, font_weight='bold', conf='{"color": "#005EB8", "opacity": 0.9}'):

        self.altair_obj = self.altair_obj.configure_axis(
            labelFontSize=label_font_size,
            titleFontSize=title_font_size
        ).configure_title(
            fontSize=font_size,
            fontWeight=font_weight
        )
        
        self.configure_mark(conf["color"], float(conf["opacity"]))

        return self
    
    def merge_charts(self, chart_list:list=[]):
        # Combine the two charts
        combined_chart = alt.layer(
            *chart_list
        ).resolve_scale(color='independent')

        return combined_chart
    
    def merge_charts_horizontally(self, chart_list:list=[]):
        # Combine the two charts
        combined_chart = alt.hconcat(
            *chart_list
        ).resolve_scale(color='independent')

        return combined_chart

        
    def merge_charts_vertically(self, chart_list:list=[]):
        # Combine the two charts
        combined_chart = alt.vconcat(
            *chart_list
        ).resolve_scale(color='independent')

        return combined_chart
    
    def configure_mark(self, color='#005EB8', opacity=1):
        self.altair_obj = self.altair_obj.configure_mark(
            opacity=opacity,
            color= color
        )
        return self
    
    def legend_config(self):
        self.altair_obj = self.altair_obj.configure_legend(
            orient='bottom',
            titleLimit=0
        )
        return self
    

    def interactive(self):
        self.altair_obj = self.altair_obj.interactive()  # Make the chart interactive

        return self

    def return_obj(self):
        return self.altair_obj
    
    def return_dict_obj(self):
        return self.altair_obj.to_dict()
    
    def show(self):
        return self.altair_obj.show()
    
    @staticmethod
    def correlation_matrix(data, variables:list = ['variable2:O', 'variable:O'], correlation_col:str = "correlation:Q", correlation_label:str = "correlation_label", display_value_text:bool = False):
        base = alt.Chart(data).encode(
            x=variables[0],
            y=variables[1]    
        )

        # Text layer with correlation labels
        # Colors are for easier readability
        if (display_value_text):
            text = base.mark_text().encode(
                text=correlation_label,
                color=alt.condition(
                    alt.datum.correlation > 0.5, 
                    alt.value('white'),
                    alt.value('black')
                )
            )

        # The correlation heatmap itself
        cor_plot = base.mark_rect().encode(
            color=correlation_col
        )

        if(display_value_text):
            chart = cor_plot + text
        else:
            chart = cor_plot

        return chart


    @staticmethod
    def outlier_visualization(df):
        # Melt the DataFrame to convert it to long format
        df_melted = df.melt(var_name='Column', value_name='Value')

        # Create a box plot with Altair
        box_plot = alt.Chart(df_melted).mark_boxplot().encode(
            x='Column:O',
            y='Value:Q'
        )

        # Create a scatter plot with Altair
        scatter_plot = alt.Chart(df_melted).mark_circle().encode(
            x='Column:O',
            y='Value:Q'
        )

        # Combine both plots
        combined_plot = box_plot + scatter_plot

        # Display the combined plot
        return combined_plot

    @staticmethod
    def plot_bar_chart(df, x_axis="", conf={}, width=0, selected_content=""):
        if("x-angle" in conf):
            labelAngle = conf["x-angle"]
        else:
            labelAngle = translateCategoricalData(x_axis)
        # Create an Altair bar chart with sorting based on 'Values'
        x_axis_acronym, _ = acronym(x_axis)  # Replace with your actual acronym function. title here is not working
        selected_content = selected_content.replace("rcsentinfo_", "")
        
        if(x_axis == "bibliography_year"):
            append_for_title = "over time" # needed for title
            X = alt.X(f'{x_axis}:O', title="Year")
            Y = alt.Y('Cumulative MP Structures:Q')
        else:
            # Y = alt.Y('Cumulative MP Structures:Q', scale=alt.Scale(type='log'))
            Y = alt.Y('Cumulative MP Structures:Q')
            append_for_title = ""
            if(sortAxis(x_axis)):
                X = alt.X(f'{x_axis}:O', axis=alt.Axis(labelFontSize=14), sort=alt.SortOrder('ascending'), title=x_axis_acronym)
            elif(sortRangeAxis(x_axis)):
                processed_resolution_list = df[x_axis].tolist()
                # Sort the list based on the start value of each range
                processed_resolution_list_sorted = sorted(processed_resolution_list, key=lambda x: float(x.split('-')[0]))
                X = alt.X(f'{x_axis}:O', sort=processed_resolution_list_sorted, title=x_axis_acronym)
                Y = alt.Y('Cumulative MP Structures:Q')
            else:
                X = alt.X(f'{x_axis}:O', axis=alt.Axis(labelFontSize=9), sort=alt.EncodingSortField(field='Cumulative MP Structures', order='descending'), title=x_axis_acronym)
        title = getTitle(selected_content)
        
        if title == "" or title == selected_content:
            title = ["Cumulative sum of resolved Membrane Protein ", "(MP) Structures categorized by " + (selected_content.split("*")[0] + " (" + selected_content.split("*")[1].upper() + ") " + append_for_title if len(selected_content.split("*")) > 1 else selected_content.split("*")[0]).replace("_", " ")]
      
        """
        if((x_axis != "bibliography_year") and not sortRangeAxis(x_axis)):
            title[-1] += " (Log Scale)"
        """
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=X,
            y=Y,
            text="Cumulative MP Structures",
            tooltip=[alt.Tooltip(tooltip, title=format_string_caps(tooltip.capitalize())) for tooltip in [x_axis, 'Cumulative MP Structures']]
        ).properties(
            width=width if width > 0 else "container",
            title=title
        )
        # Create the mark_text layer
        text_layer = bar_chart.mark_text(align='center', dy=-10)
        # Layer the bar chart and mark_text layer
        layered_chart = alt.layer(bar_chart, text_layer)

        # Define the configuration for the entire chart

        layered_chart = layered_chart.configure_axis(
            labelAngle=labelAngle,  # Adjust the angle as needed
            labelLimit=0,
        ).configure_legend(orient='bottom').configure_mark(
            opacity=float(conf["opacity"] if conf and conf["opacity"] else 0.9),
            color=conf["color"] if conf and conf["color"] else "#005EB8"
        )

        # Return the layered chart
        return layered_chart
    
def sortAxis(axis:str):
    axis_list = [
        "taxonomic_domain"
    ]
    return axis in axis_list


def sortRangeAxis(axis:str):
    axis_list = [
        "resolution", "processed_resolution", "rcsentinfo_molecular_weight", "rcsentinfo_deposited_atom_count",
    ]
    return axis in axis_list

 
def acronym(input_string):
    input_string = input_string.replace("rcsentinfo_", "")
    input_replacement, title = replace_value(input_string)
    words = input_replacement.split("_")
    acronym = ' '.join(word.capitalize() for word in words)
    if input_string == "processed_resolution":
        acronym = "Resolution (Angstroms (Å))"
    if input_string == "exptl_crystal_grow_method1":
        acronym = "Crystal Grow Method"
    return acronym, title


def translateCategoricalData(x_axis):
    columns = [
        "species", "resolution", "processed_resolution",
        "rcsentinfo_software_programs_combined", 
        "symspagroup_name_hm", "rcsentinfo_molecular_weight", 
        "rcsentinfo_deposited_atom_count", "bibliography_year", 
        "expressed_in_species", "rcspricitation_rcsb_journal_abbrev",
        "exptl_crystal_grow_method1", "exptl_crystal_grow_method2"
    ]
    labelAngle = 360
    if(x_axis in columns):
        labelAngle = -50
    return labelAngle

def getTitle(key):
    _, title = replace_value(key)
    return title

def replace_value(value):
    """
    Replace a value based on a dictionary of replacements.

    Parameters:
        replacement_dict (dict): A dictionary where keys are values to be replaced
                                 and values are their replacements.
        value (str): The value to check for replacement.

    Returns:
        str: The replaced value if found in the replacement dictionary,
             otherwise returns the original value.
    """
    replacement_dicts = [
        { "symspagroup_name_hm": "Space Group", "title": ["Cumulative sum of resolved Membrane Protein (MP) ", "Structures categorized by Space Group."]},
        { 
            "expressed_in_species*e._coli": "Expressed in Species", 
            "title": ["Cumulative sum of resolved Membrane Protein (MP) Structures ", "categorized by their expression in the species (E. coli.)"]
        },
        { 
            "expressed_in_species*s._frugiperda": "Expressed in Species", 
            "title": ["Cumulative sum of resolved Membrane Protein (MP) Structures ", "categorized by their expression in the species (S. frugiperda)"]
        },
        { 
            "expressed_in_species*hek293_cells": "Expressed in Species", 
            "title": ["Cumulative sum of resolved Membrane Protein (MP) Structures ", "categorized by their expression in the species ((HEK) 293)"]
        },
        {
            "processed_resolution": "Resolution (Angstroms (Å))",
            "title": ["Cumulative range distribution of resolved Membrane Protein", " (MP) Structures categorized by resolution."]
        },
        {
            "exptl_crystal_grow_method1": "Crystal Growth Method",
            "title": [
                "Cumulative sum of resolved Membrane Protein (MP) Structures", 
                " categorized by crystal growth method."
            ]
        }
    ]
    for replacement_dict in replacement_dicts:
        if value in replacement_dict:
            return replacement_dict[value], replacement_dict.get('title')
    return value, value