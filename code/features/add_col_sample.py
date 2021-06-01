import os
from typing import Dict, List
import pandas as pd

def run(df: pd.DataFrame, using_cache:bool)->List[Dict]:
    new_column = df['assessmentItemID'].apply(lambda x : x[:3])
    new_column.name = "grade"
    # Category type 데이터는 dtype이 object여야 합니다.
    if new_column.dtype != object: new_column = new_column.astype(str)

    new_column2 = df.groupby('userID')['answerCode'].transform('mean')
    new_column2.name = "user_acc"
    # Continuous 데이터는 float type, Discrete 데이터는 int type이여야 합니다.
    if new_column2.dtype == object: raise TypeError("continuous data shouldn't be object.")

    # 여러 columns을 추가시, columns 내부의 list로 나열해주면 됩니다.
    # 하지만, job이 하나만 존재하여도 list로 감싸줘야합니다. 
    return [{
        "job": "add",
        "columns":[
            new_column,
            new_column2
        ]
    }]