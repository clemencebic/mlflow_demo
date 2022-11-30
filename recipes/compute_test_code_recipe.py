# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import math

# Read recipe inputs
eval_data = dataiku.Dataset("eval_data")
eval_data_df = eval_data.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

test_code_recipe_df = eval_data_df # For this sample code, simply copy input to output


# Write recipe outputs
test_code_recipe = dataiku.Dataset("test_code_recipe")
test_code_recipe.write_with_schema(test_code_recipe_df)
