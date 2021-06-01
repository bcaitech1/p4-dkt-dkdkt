import os
from typing import Dict, List
import pandas as pd

def run(df: pd.DataFrame, using_cache:bool)->List[Dict]:
    new_column1 = df['assessmentItemID'].apply(lambda x : x[:3])
    new_column2 = df['assessmentItemID'].apply(lambda x : x[3:])
    new_column1.name = "grade"
    new_column2.name = "problemID"

    # 다른 성격의 job의 경우(ex) add <=> del), return 내부의 array에 따로 나열해줘야합니다.
    return [
        {
            "job": "add",
            "columns":[
                new_column1, new_column2
            ]
        },
        {
            "job": "del",
            "columns":[
                "assessmentItemID"
            ]
        },
    ]