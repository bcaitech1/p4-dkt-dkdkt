import os
import pandas as pd
'''
1. add colmun
2. delete column
3. change column
4. change sequence
5. add row
'''

def get_cache(colmun_name:str, data_dir='/opt/ml/input/data/train_dataset/features/', parse_dates=False, dtype=None):    
    if parse_dates:
        return pd.read_csv(data_dir+colmun_name, dtype=dtype, parse_dates=[colmun_name])
    else:
        return pd.read_csv(data_dir+colmun_name, dtype=dtype)


def add_feature(df:pd.DataFrame, colmun_name:str, use_cache=True, suffix='', data_dir='/opt/ml/input/data/train_dataset/features/') -> pd.DataFrame:
    if use_cache:
        if os.path.exists(data_dir+colmun_name+suffix):
            column = get_cache(colmun_name,data_dir)
    '''

    '''
    raise NotImplementedError(f"NotImplemetedError for {colmun_name}")
    if not use_cache or not column:
        colmun = df['']
    
    

    if use_cache:
        column.to_csv(data_dir+colmun_name+suffix, index=False)

    df[colmun_name] = column
    return df
    