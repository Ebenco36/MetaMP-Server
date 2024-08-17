# Standard imports
import pandas as pd
import numpy as np
import altair as alt


from vega_datasets import data
cars = data.cars()
print(cars.sample(5))
import altair as alt
alt.Chart(cars).mark_point()
alt.Chart(cars).mark_point().encode(x="Miles_per_Gallon", y="Horsepower")