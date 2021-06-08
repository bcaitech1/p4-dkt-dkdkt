import os
from datetime import datetime
import time
import tqdm
import pickle
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import math 
import copy 

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
    

    def get_train_data(self):
        return self.train_data


    def get_test_data(self):
        return self.test_data


    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2


    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)


    def __preprocessing(self, df, is_train = True):
        # TODO 2 : args에 따라 카테고리컬 변수를 정할 수 있게 하기
        cate_cols = self.args.cate_col
        # cont_cols = self.args.cont_col

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        # def convert_time(s):
        #     timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
        #     return int(timestamp)

        # df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df


    def __feature_engineering(self, df):
        # TODO 1 : continuous & categorical 변수 추가하기
        data_path = os.path.join(self.args.asset_dir,f"train_data_FE.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            # with open(data_path,'rb') as file:
            #     df=pickle.load(file)
        else :
            df.sort_values(by=['userID','Timestamp'], inplace=True)
            
            df['hour'] = df['Timestamp'].dt.hour
            df['dow'] = df['Timestamp'].dt.dayofweek
            
            diff = df.loc[:, ['userID','Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
            diff = diff.fillna(pd.Timedelta(seconds=0))
            diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

            # 푸는 시간
            df['elapsed'] = diff
            df['elapsed'] = df['elapsed'].apply(lambda x : x if x <650 and x >=0 else 0)
            
            df['grade']=df['testId'].apply(lambda x : int(x[1:4])//10)
            df['mid'] = df['testId'].apply(lambda x : int(x[-3:]))
            df['problem_number'] = df['assessmentItemID'].apply(lambda x : int(x[-3:]))
            
            correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
            correct_t.columns = ["test_mean", 'test_sum']
            correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
            correct_k.columns = ["tag_mean", 'tag_sum']
            correct_a = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
            correct_a.columns = ["ass_mean", 'ass_sum']
            correct_p = df.groupby(['problem_number'])['answerCode'].agg(['mean', 'sum'])
            correct_p.columns = ["prb_mean", 'prb_sum']
            correct_h = df.groupby(['hour'])['answerCode'].agg(['mean', 'sum'])
            correct_h.columns = ["hour_mean", 'hour_sum']
            correct_d = df.groupby(['dow'])['answerCode'].agg(['mean', 'sum'])
            correct_d.columns = ["dow_mean", 'dow_sum'] 
            
            df = pd.merge(df, correct_t, on=['testId'], how="left")
            df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
            df = pd.merge(df, correct_a, on=['assessmentItemID'], how="left")
            df = pd.merge(df, correct_p, on=['problem_number'], how="left")
            df = pd.merge(df, correct_h, on=['hour'], how="left")
            df = pd.merge(df, correct_d, on=['dow'], how="left")

            o_df = df[df['answerCode']==1]
            x_df = df[df['answerCode']==0]
            
            elp_k = df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
            elp_k.columns = ['KnowledgeTag',"tag_elp"]
            elp_k_o = o_df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
            elp_k_o.columns = ['KnowledgeTag', "tag_elp_o"]
            elp_k_x = x_df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
            elp_k_x.columns = ['KnowledgeTag', "tag_elp_x"]
            
            df = pd.merge(df, elp_k, on=['KnowledgeTag'], how="left")
            df = pd.merge(df, elp_k_o, on=['KnowledgeTag'], how="left")
            df = pd.merge(df, elp_k_x, on=['KnowledgeTag'], how="left")

            ass_k = df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
            ass_k.columns = ['assessmentItemID',"ass_elp"]
            ass_k_o = o_df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
            ass_k_o.columns = ['assessmentItemID',"ass_elp_o"]
            ass_k_x = x_df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
            ass_k_x.columns = ['assessmentItemID',"ass_elp_x"]

            df = pd.merge(df, ass_k, on=['assessmentItemID'], how="left")
            df = pd.merge(df, ass_k_o, on=['assessmentItemID'], how="left")
            df = pd.merge(df, ass_k_x, on=['assessmentItemID'], how="left")

            prb_k = df.groupby(['problem_number'])['elapsed'].agg('mean').reset_index()
            prb_k.columns = ['problem_number',"prb_elp"]
            prb_k_o = o_df.groupby(['problem_number'])['elapsed'].agg('mean').reset_index()
            prb_k_o.columns = ['problem_number',"prb_elp_o"]
            prb_k_x = x_df.groupby(['problem_number'])['elapsed'].agg('mean').reset_index()
            prb_k_x.columns = ['problem_number',"prb_elp_x"]

            df = pd.merge(df, prb_k, on=['problem_number'], how="left")
            df = pd.merge(df, prb_k_o, on=['problem_number'], how="left")
            df = pd.merge(df, prb_k_x, on=['problem_number'], how="left")
            
            df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
            df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
            df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
            df['Grade_o'] = df.groupby(['userID','grade'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
            df['GradeCount'] = df.groupby(['userID','grade']).cumcount()
            df['GradeAcc'] = df['Grade_o']/df['GradeCount']
            df['GradeElp'] = df.groupby(['userID','grade'])['elapsed'].transform(lambda x: x.cumsum().shift(1))
            df['GradeMElp'] = df['GradeElp']/df['GradeCount']
            
            f = lambda x : len(set(x))
            test_df = df.groupby(['testId']).agg({
                'problem_number':'max',
                'KnowledgeTag':f
            })
            test_df.reset_index(inplace=True)

            test_df.columns = ['testId','problem_count',"tag_count"]
            
            df = pd.merge(df,test_df,on='testId',how='left')
            
            gdf = df[['userID','testId','problem_number','grade','Timestamp']].sort_values(by=['userID','grade','Timestamp'])
            gdf['buserID'] = gdf['userID'] != gdf['userID'].shift(1)
            gdf['bgrade'] = gdf['grade'] != gdf['grade'].shift(1)
            gdf['first'] = gdf[['buserID','bgrade']].any(axis=1).apply(lambda x : 1- int(x))
            gdf['RepeatedTime'] = gdf['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)) 
            gdf['RepeatedTime'] = gdf['RepeatedTime'].apply(lambda x: x.total_seconds()) * gdf['first']
            df['RepeatedTime'] = gdf['RepeatedTime'].apply(lambda x : math.log(x+1))
            
            df['prior_KnowledgeTag_frequency'] = df.groupby(['userID','KnowledgeTag']).cumcount()
            
            df['problem_position'] = df['problem_number'] / df["problem_count"]
            df['solve_order'] = df.groupby(['userID','testId']).cumcount()
            df['solve_order'] = df['solve_order'] - df['problem_count']*(df['solve_order'] > df['problem_count']).apply(int) + 1
            df['retest'] = (df['solve_order'] > df['problem_count']).apply(int)
            T = df['solve_order'] != df['problem_number']
            TT = T.shift(1)
            TT[0] = False
            df['solved_disorder'] = (TT.apply(lambda x : not x) & T).apply(int)
            
            df['testId'] = df['testId'].apply(lambda x : int(x[1:4]+x[-3]))
            df['hour'] = df['Timestamp'].dt.hour
            df['dow'] = df['Timestamp'].dt.dayofweek
            
            df = df.fillna(0)
            # with open(data_path,'wb') as file:
            #     pickle.dump(df,file)
            df.to_csv(data_path)
        return df


    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])#, nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

                
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        self.args.n_cont = len(self.args.cont_col)
        # TODO 4 : 연속형 변수 임베딩 크기 저장하기


        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        # TODO 5 : args에 따라 column 및 group 설정할 수 있게 하기
        # columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag']
        columns = self.args.cate_col + self.args.cont_col + ['userID','answerCode']
        
        group = df[columns].groupby('userID').apply(lambda row : 
            tuple(row[c].values for c in row.columns if not c == 'userID')
            )
        # group = df[columns].groupby('userID').apply(
        #         lambda r: (
        #             r['testId'].values, 
        #             r['assessmentItemID'].values,
        #             r['KnowledgeTag'].values,
        #             r['answerCode'].values
        #         )
        #     )
        return group.values


    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)


    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        # TODO 6 : 정해지지 않은 변수로 받을 수 있게 하기 
        
        cate_cols = [row[i] for i in range(0,len(row))]
        # cate_cols = [test, question, tag, correct]
        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

        
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)


    for i, _ in enumerate(col_list):
        col_list[i] =torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None
    
    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader

def add_features(args):
    cate_dict = {
        'base' : ['assessmentItemID', 'testId', 'KnowledgeTag'],
        'cont' : ['assessmentItemID', 'testId', 'KnowledgeTag']
    }
    cont_dict = {
        'base' : [],
        'cont' : ['GradeAcc','GradeMElp','user_acc',
                  'RepeatedTime','GradeElp','user_correct_answer',
                  'elapsed','ass_mean','user_total_answer','ass_elp_o',
                  "ass_elp","GradeCount"]
    }
    args.cate_col = cate_dict[args.feature_type]
    args.cont_col = cont_dict[args.feature_type]
    print("cate_columns : ", args.cate_col)
    print("cont_columns : ", args.cont_col)
    return args