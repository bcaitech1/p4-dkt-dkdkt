import os
import pandas as pd
'''
suffix of preprocessing
- a: add row
- b: change sequence
- c: add colmun
- d: change column
- e: delete column
'''

def get_cache(colmun_name:str, data_dir='/opt/ml/input/data/train_dataset/features/', parse_dates=False, dtype=None):    
    if parse_dates:
        return pd.read_csv(data_dir+colmun_name, dtype=dtype, parse_dates=[colmun_name])
    else:
        return pd.read_csv(data_dir+colmun_name, dtype=dtype)


def a_add_feature(df:pd.DataFrame, colmun_name:str, use_cache=True, suffix='', data_dir='/opt/ml/input/data/train_dataset/features/') -> pd.DataFrame:
    column = None
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

def a_add_testset(train_df:pd.DataFrame, test_df_dir='/opt/ml/input/data/train_dataset/test_data.csv'):
    print("add testset")
    test_df = pd.read_csv(test_df_dir, parse_dates=['Timestamp'])
    test_df = test_df[test_df['answerCode'] != -1]
    result = train_df.append(test_df, ignore_index=True)
    return result

def c_add_grade(df:pd.DataFrame):
    print("add grade")
    df['grade'] = df['assessmentItemID'].apply(lambda x : x[:3])
    return df