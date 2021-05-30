import os
import pandas as pd

def add_cat_col_sample(df:pd.DataFrame):
    print("add grade")
    df['grade'] = df['assessmentItemID'].apply(lambda x : x[:3])
    return {
        "job": "add_cat",
        
    }