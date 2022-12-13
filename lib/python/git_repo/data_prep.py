import pandas as pd
import numpy as np

def compute_sum_of_col(df, col1, col2):
    df['new_sum_col'] = df[col1] + df[col2]
    return 

def compute_substr_of_col(df, col1, col2):
    df['new_substr_col'] = df[col1] - df[col2]
    return df

def compute_product_of_col(df, col1, col2):
    df['new_product_col'] = df[col1] * df[col2]
    return df

def compute_div_of_col(df, col1, col2):
    df['new_product_col'] = df[col1] / df[col2]
    return df

def compute_div_of_col_b(df, col1, col2):
    df['new_product_col'] = df[col1] / df[col2]
    return df

def sum(a,b):
    return a+b
