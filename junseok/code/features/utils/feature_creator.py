import os
import pandas as pd
import json
from tqdm.auto import tqdm
import time
from datetime import datetime

def convert_time(s):
    timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
    return int(timestamp)
    
def feature_creator(exp_dir:str, col_name:str, data_dict:dict):
    # 공유, FE를 위한 Column을 생성하기 위한 function
    # 총, train, validation, test dataset의 column을 만들고 저장함.

    # user_acc 예시를 위한 전체 dataset
    # train_df = pd.read_csv('/opt/ml/input/data/train_dataset/train_cached.csv')
    # user_acc = train_df.groupby('userID')['answerCode'].mean()

    for mode, csv_file in tqdm(data_dict.items(), desc="processing features..."):
        df = pd.read_csv(csv_file)
        # df['temp_idx'] = df.index # 인덱스를 인위적으로 바꾸었을 경우 인덱스 정렬 대신 사용

        # Timestamp를 사용해야할 경우의 코드
        # if df['Timestamp'].dtype == object or df['Timestamp'].dtype == "datetime64[ns]":
        #     # Timestamp 전처리가 이미 되어있는 경우 전처리가 실행되지 않음.
        #     df['Timestamp'] = df['Timestamp'].apply(convert_time)


        # if mode == "test": 
            # test의 경우 다른 전처리가 필요한 경우의 분기 코드
            # (ex) test dataset의 경우, answerCode가 일부는 -1로 마스킹되어 있음. 이를 처리하기 위한 다른 코드 필요)
        # else:       
 
        ############################# YOUR CODE HERE ########################
        #                                                                   #
        #                                                                   #
        #                                                                   #
        # # elapsed_time 예시                                               #
        # df = df.sort_values(by=['userID', 'Timestamp'], axis=0)           #
        # df[col_name] = df['Timestamp'].diff().fillna(0)                   #
        #                                                                   #
        # # user_acc 예시                                                   #
        # df['user_acc'] = train_df['userID'].apply(lambda x: user_acc[x])  #
        #                                                                   #
        #                                                                   #
        #                                                                   # 
        ############################# Fill the code #########################  

        df = df.sort_index() # row 순서가 바뀌었을 경우를 위한 원래대로의 정렬
        # df = df.sort_values(by=['temp_idx'], axis=0) # 인덱스를 인위적으로 바꾸었을 경우 위 코드 대신 사용.
        directory = os.path.join(exp_dir,col_name)
        os.makedirs(directory, exist_ok=True)

        # 굳이 dataframe에 column으로 만들지않고, 바로 csv로 보낼경우 바꿔주자!
        df[col_name].to_csv(os.path.join(directory,f"{mode}.csv"), index=False)

    with open(os.path.join(directory,"feature_desc.json"),"w") as f:
        data_dict["feature_name"] = col_name
        json.dump(data_dict,f, indent=4)
    print("feature exporting done!")

# base가 되는 데이터셋의 directory 표시
data_dict = {
    "train": '/opt/ml/input/data/train_dataset/cv_train_cached.csv',
    "val": '/opt/ml/input/data/train_dataset/cv_val_cached.csv',
    "test": '/opt/ml/input/data/train_dataset/test_data.csv',
}
feature_creator('/opt/ml/input/data/features/', "elapsed_time", data_dict)

# user_acc 예시 
# feature_creator('/opt/ml/input/data/features/', "user_acc", data_dict)