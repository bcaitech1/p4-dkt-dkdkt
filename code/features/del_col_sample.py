import os
from typing import Dict, List
import pandas as pd

def run(df: pd.DataFrame, _:bool)->List[Dict]:
    # 단순삭제 FE의 경우 캐쉬가 존재하지 않음. 
    # 만약, 이미 column이 존재하지 않은 경우, 무시합니다.
    return [{
        "job": "del",
        "columns":[
            'KnowledgeTag'
        ]
    }]