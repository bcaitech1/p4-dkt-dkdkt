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
import random
from multiprocessing import Pool
import multiprocessing
from collections import Counter
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
        if self.args.cv_strategy :
            if self.args.sep_grade :
                # valid_user_path = os.path.join(self.args.data_dir,'cv_strategy',"cv_train_2.pkl")
                valid_user_path = os.path.join(self.args.data_dir,"cv_valid_index.pickle")
                if os.path.exists(valid_user_path):
                    with open(valid_user_path,"rb") as file :
                        valid_idx = pickle.load(file)
                data_1 = data[~data['userID'].isin(valid_idx)]
                data_2 = data[data['userID'].isin(valid_idx)]
            else :
                valid_user_path = os.path.join(self.args.data_dir,"cv_valid_index.pickle")
                if os.path.exists(valid_user_path):
                    with open(valid_user_path,"rb") as file :
                        valid_idx = pickle.load(file)
                data_1 = data[~data['userID'].isin(valid_idx)]
                data_2 = data[data['userID'].isin(valid_idx)]
        # else :
        #     train_user_path = os.path.join(self.args.data_dir,"cv_train_index.pickle")
        #     valid_user_path = os.path.join(self.args.data_dir,"cv_valid_index.pickle")
        #     with open(train_user_path,"rb") as file :
        #         train_idx = pickle.load(file)
        #     with open(valid_user_path,"rb") as file :
        #         valid_idx = pickle.load(file)
        #     data_1 = data[data['userID'].isin(train_idx)]
        #     data_2 = data[data['userID'].isin(valid_idx)]
        else :
            idx_list = list(set(data['userID']))
            size = int(len(idx_list) * ratio)
            train_idx = random.sample(idx_list,size)
            data_1 = data[data['userID'].isin(train_idx)]
            data_2 = data[~data['userID'].isin(train_idx)]

        return data_1, data_2


    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)


    def __preprocessing(self, df, is_train = True):

        # Encoding the Categorical Embedding
        cols = list(set(self.args.cate_col + self.args.temp_col)) # not to encode twice 

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
        other = []
        for col in cols:
            le = LabelEncoder()
            if col in ['assessmentItemID', 'testId', 'KnowledgeTag', 'grade', 'last_problem', 'problem_number'] :
                if is_train:
                    #For UNKNOWN class
                    a = df[col].unique().tolist()
                    a = sorted(a) if str(type(a[0])) == "<class 'int'>" else a 
                    a = a + ['unknown']
                    le.fit(a)
                    self.__save_labels(le, col)
                else:
                    label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                    le.classes_ = np.load(label_path)
                    df[col]= df[col].astype(str)
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
                            #모든 컬럼이 범주형이라고 가정
                df[col]= df[col].astype(str)
                test = le.transform(df[col])
                df[col] = test
            else : 
                if is_train :
                    unq = df[col].unique().tolist()
                    unq = sorted(unq) if str(type(unq[0])) == "<class 'int'>" else unq 
                    other += list(map(lambda x : col+'_'+str(x),unq))
                    df[col] = df[col].apply(lambda x : col+'_'+str(x))
                else : 
                    label_path = os.path.join(self.args.asset_dir,'other_classes.npy')
                    le.classes_ = np.load(label_path)
                    df[col]= df[col].astype(str)
                    df[col] = df[col].apply(lambda x : col+'_'+x)
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

        if other :
            other += ['unknown']
            le = LabelEncoder()
            le.fit(other)
            self.__save_labels(le, 'other')

        label_path = os.path.join(self.args.asset_dir,'other_classes.npy')
        if os.path.exists(label_path):
            le.classes_ = np.load(label_path)

            for col in cols:
                if col in ['assessmentItemID', 'testId', 'KnowledgeTag', 'grade', 'last_problem', 'problem_number'] :
                    continue
                else : 
                    df[col]= df[col].astype(str)
                    test = le.transform(df[col])
                    df[col] = test

        if not is_train and self.args.sep_grade:
            ddf = df[df['answerCode']==-1]
            df = df[df.set_index(['userID','grade']).index.isin(ddf.set_index(['userID','grade']).index)]
        
        return df


    def __feature_engineering(self, df,file_name, is_train):

        data_path = os.path.join(self.args.asset_dir,f"{file_name[:-4]}_FE.pkl") # .csv빼고 
        if os.path.exists(data_path):
            df = pd.read_pickle(data_path)

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
            
            if is_train : 
                sub_data_path = os.path.join("/opt/ml/input/data/train_dataset/test_data.csv") # .csv빼고 
                sub_df = pd.read_csv(sub_data_path)
                full_df = pd.concat([df,sub_df[sub_df['answerCode']!= -1]])
            else :
                sub_data_path = os.path.join(self.args.asset_dir,"train_data_FE.csv") # .csv빼고 
                sub_df = pd.read_csv(sub_data_path)
                full_df = pd.concat([df[df['answerCode']!= -1],sub_df])
                
            correct_t = full_df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
            correct_t.columns = ["test_mean", 'test_sum']
            correct_k = full_df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
            correct_k.columns = ["tag_mean", 'tag_sum']
            correct_a = full_df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
            correct_a.columns = ["ass_mean", 'ass_sum']
            correct_p = full_df.groupby(['problem_number'])['answerCode'].agg(['mean', 'sum'])
            correct_p.columns = ["prb_mean", 'prb_sum']
            correct_h = full_df.groupby(['hour'])['answerCode'].agg(['mean', 'sum'])
            correct_h.columns = ["hour_mean", 'hour_sum']
            correct_d = full_df.groupby(['dow'])['answerCode'].agg(['mean', 'sum'])
            correct_d.columns = ["dow_mean", 'dow_sum'] 
            
            df = pd.merge(df, correct_t, on=['testId'], how="left")
            df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
            df = pd.merge(df, correct_a, on=['assessmentItemID'], how="left")
            df = pd.merge(df, correct_p, on=['problem_number'], how="left")
            df = pd.merge(df, correct_h, on=['hour'], how="left")
            df = pd.merge(df, correct_d, on=['dow'], how="left")

            o_df = full_df[full_df['answerCode']==1]
            x_df = full_df[full_df['answerCode']==0]
            
            elp_k = full_df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
            elp_k.columns = ['KnowledgeTag',"tag_elp"]
            elp_k_o = o_df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
            elp_k_o.columns = ['KnowledgeTag', "tag_elp_o"]
            elp_k_x = x_df.groupby(['KnowledgeTag'])['elapsed'].agg('mean').reset_index()
            elp_k_x.columns = ['KnowledgeTag', "tag_elp_x"]
            
            df = pd.merge(df, elp_k, on=['KnowledgeTag'], how="left")
            df = pd.merge(df, elp_k_o, on=['KnowledgeTag'], how="left")
            df = pd.merge(df, elp_k_x, on=['KnowledgeTag'], how="left")

            ass_k = full_df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
            ass_k.columns = ['assessmentItemID',"ass_elp"]
            ass_k_o = o_df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
            ass_k_o.columns = ['assessmentItemID',"ass_elp_o"]
            ass_k_x = x_df.groupby(['assessmentItemID'])['elapsed'].agg('mean').reset_index()
            ass_k_x.columns = ['assessmentItemID',"ass_elp_x"]

            df = pd.merge(df, ass_k, on=['assessmentItemID'], how="left")
            df = pd.merge(df, ass_k_o, on=['assessmentItemID'], how="left")
            df = pd.merge(df, ass_k_x, on=['assessmentItemID'], how="left")

            prb_k = full_df.groupby(['problem_number'])['elapsed'].agg('mean').reset_index()
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
            test_df = full_df.groupby(['testId']).agg({
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
            df['last_problem'] = (df['testId']!=df['testId'].shift(-1)).apply(int)
            # df = ELO_function(df)

            df = df.fillna(0)
            # with open(data_path,'wb') as file:
            #     pickle.dump(df,file)
            df.to_pickle(data_path)
        return df


    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])#, nrows=100000)
        df = self.__feature_engineering(df,file_name, is_train)
        df = self.__preprocessing(df, is_train)
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        self.args.n_grade = len(np.load(os.path.join(self.args.asset_dir,'grade_classes.npy')))
        self.args.n_other = len(np.load(os.path.join(self.args.asset_dir,'other_classes.npy')))
        self.args.n_cont = len(self.args.cont_col)
        df = df.sort_values(by=self.args.key, axis=0)
        return df


    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)


    def load_test_data(self, file_name, is_train_):
        self.test_data = self.load_data_from_file(file_name, is_train= is_train_)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args, augment):
        self.data = data
        self.args = args
        self.augment = augment

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        do_augment = random.random()
        try :     
            if self.augment and do_augment <self.args.augment_rate: 
                columns = self.args.cate_col + self.args.cont_col + self.args.temp_col
                l_idx = columns.index('last_problem')
                nx = torch.Tensor(row[l_idx]).numpy()
                last_problems = np.where(nx==max(nx))[0]
                p = np.array([self.test_dist(x) for x in last_problems])
                p = p/sum(p)
                idx = np.random.choice(len(last_problems),1,p=p)[0]
                seq_len = last_problems[idx] if last_problems[idx] > 0 else seq_len
        except :
            print('augmented error in get_item')
            pass
        
        cols = [row[i][:seq_len] for i in range(0,len(row))]
        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cols):
                cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cols):
            cols[i] = torch.tensor(col)

        return cols

    def __len__(self):
        return len(self.data)

    def test_dist(self,x):
        if self.args.sep_grade == True :
            # pmf = [888,293,171,142,130,82,71,57,38,39,27,19,15,7,1,3,1,3,1]
            pmf = [3,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,7,7,7,7,8]
            x = pmf[abs(x)//50 if x < 950 else 18]
        else :
            pmf = [229,108,76,62,51,42,41,28,24,14,4,9,1,1,1]
            x = pmf[abs(x)//100 if x < 1700 else 16]
        return x

from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])
    try : 
        # batch의 값들을 각 column끼리 그룹화
        for row in batch:
            for i, col in enumerate(row):
                pre_padded = torch.zeros(max_seq_len)
                pre_padded[-len(col):] = col
                col_list[i].append(pre_padded)
    except :
        print("augment error")


    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None
    
    if train is not None:
        trainset = DKTDataset(train, args, args.augment)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args, False)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader

def add_features(args):

    cate_dict = {
        'base' : ['assessmentItemID', 'testId', 'KnowledgeTag'],
        'cont' : ['assessmentItemID', 'testId', 'KnowledgeTag','grade'],
        'ver1' : ['assessmentItemID', 'mid', 'KnowledgeTag','grade'],
        'ver2' : ['assessmentItemID', 'KnowledgeTag','grade'],
        'ver3' : ['assessmentItemID', 'testId', 'KnowledgeTag','grade','retest'],

    }

    cont_dict = {
        'base' : [],
        'cont' : ['test_mean', 'prb_mean', 'GradeAcc', 'user_acc', 
                    'GradeMElp', 'RepeatedTime', 'GradeElp', 'ass_mean', 'GradeCount', 
                    'tag_mean', 'prb_elp', 'elapsed','user_correct_answer', 'user_total_answer'],
        'ver1' : ['test_mean', 'prb_mean', 'prb_elp_x', 'GradeAcc', 'user_acc', 
                    'GradeMElp', 'RepeatedTime', 'GradeElp', 'ass_mean', 'GradeCount', 
                    'tag_mean', 'prb_elp', 'elapsed', 'prb_elp_o','ass_elp', 'ass_elp_o', 'ass_elp_x',
                    'user_correct_answer', 'user_total_answer'
                    ],
        'ver2' : ['test_mean', 'prb_mean', 'GradeAcc', 'user_acc', 
                    'GradeMElp', 'RepeatedTime', 'GradeElp', 'ass_mean', 'GradeCount', 
                    'tag_mean', 'prb_elp', 'elapsed','user_correct_answer', 'user_total_answer'],
        'ver3' : ['test_mean', 'prb_mean', 'GradeAcc', 'user_acc', 
                    'GradeMElp', 'RepeatedTime', 'GradeElp', 'ass_mean', 'GradeCount', 
                    'tag_mean', 'prb_elp', 'elapsed','user_correct_answer', 'user_total_answer'],
    }
    
    saint_cate = {
        'base' : ['grade', 'KnowledgeTag','assessmentItemID','testId'],
        'ver1' : ['grade', 'KnowledgeTag','assessmentItemID','mid'],
        'ver2' : []
    }
    saint_cont = {
        'base_d' : ["elapsed","GradeMElp","GradeAcc","user_acc","RepeatedTime","user_correct_answer","user_total_answer"],
        'ver1_d' : ["elapsed","GradeMElp","GradeAcc","user_acc","RepeatedTime","user_correct_answer","user_total_answer","test_delta","ass_delta","tag_delta"],
        'ver2_d' : ["elapsed","GradeMElp","GradeAcc","user_acc","RepeatedTime"],
        'base_e' : ["ass_elp","ass_elp_o","ass_elp_x","prb_elp","prb_elp_o","prb_elp_x","test_mean","ass_mean","test_mean","prb_mean"],
        'ver1_e' : ["ass_elp","ass_elp_o","ass_elp_x","prb_elp","prb_elp_o","prb_elp_x","tag_mean","ass_mean","test_mean","prb_mean","test_delta","ass_delta","tag_delta"],
        'ver2_e' : ["ass_elp","ass_elp_o","ass_elp_x","prb_elp","prb_elp_o","prb_elp_x","tag_mean","ass_mean","test_mean","prb_mean","elo_prob"],
    }
    
    
    args.cate_col_e = saint_cate[args.cate_col_e_type]
    args.cate_col_d = saint_cate[args.cate_col_d_type]
    args.cont_col_e = saint_cont[args.cont_col_e_type]
    args.cont_col_d = saint_cont[args.cont_col_d_type]
    
    args.n_cate_e = len(args.cate_col_e)
    args.n_cate_d = len(args.cate_col_d)
    args.n_cont_e = len(args.cont_col_e)
    args.n_cont_d = len(args.cont_col_d)
    
    if args.model == 'saint':
        args.cate_col = list(set(args.cate_col_e + args.cate_col_d))
        args.cont_col = list(set(args.cont_col_e + args.cont_col_d))
    else :
        args.cate_col = list(set(cate_dict[args.feature_type])) # need to check seperate cate and cont
        args.cont_col = list(set(cont_dict[args.feature_type]))
    
    args.temp_col = []    
    args.key = ['userID','grade','Timestamp'] if args.sep_grade else ['userID', 'Timestamp']

    if args.sep_grade == True and args.cate_col.count('grade') == 0 :
        args.temp_col.append('grade')

    if args.augment == True and args.cate_col.count('last_problem') == 0 :
        args.temp_col.append('last_problem')    

    if args.interaction_type == 'problem_number' and args.cate_col.count('problem_number') == 0:
        args.temp_col.append('problem_number')

    print("cate_columns : ", args.cate_col)
    print("cont_columns : ", args.cont_col)
    print("temp_columns : ", args.temp_col)
    print("loss_type : ", args.loss_type)

    return args

def post_process(df, args):  
    df = df.sort_values(by=args.key, axis=0)
    columns = args.cate_col + args.cont_col + args.temp_col + ['userID','answerCode']
    group = df[columns].groupby(args.key[:-1]).apply(lambda row :
            tuple(row[c].values for c in row.columns if not c == 'userID')
            )
    
    return group.values

def sweep_apply(args, config):

    args.cont_col_e_type = config.cont_col_e_type
    args.cont_col_d_type = config.cont_col_d_type

    return args

# def slidding_window(df,args):
#     window_size = args.max_seq_len
#     stride = window_size
#     augmented = []
#     for row in data:
#         seq_len = len(row[0])
#         if seq_len <= window_size :
#             augmented_datas.append(row)
#         else :
#             how_many = int((seq_len-window_size)/stride) + 1
            
#             for i in range(how_many):
#                 window_data = [col[i*stride:i*stride+window_size] for col in row]
            
                
def slide_window(data, how_many=3, ty='train', args=None):
    augmented = []
    columns = args.cate_col + args.cont_col + args.temp_col + ['answerCode']
    l_idx = columns.index('last_problem')
    if ty == 'train':
        for row in data :
            seq_len = len(row[0])
            if seq_len < how_many * 10 :
                augmented.append(row)
            else :
                nx = torch.Tensor(row[l_idx]).numpy()
                last_problems = np.where(nx==max(nx))[0]
                n_test = len(last_problems)
                unit = n_test//how_many
                r = n_test % how_many
                for i in range(1,how_many+1):
                    c_len = last_problems[unit*i+r-1 if unit*i+r-1 < n_test else n_test-1]
                    augmented.append([col[:c_len] for col in row])
    else :
        for row in data :
            seq_len = len(row[0])
            nx = torch.Tensor(row[l_idx]).numpy()
            last_problems = np.where(nx==max(nx))[0]
            n_test = len(last_problems)
            unit = n_test//how_many
            for i in range(1,min(n_test,how_many)+1):
                c_len = last_problems[-i]
                augmented.append([col[:c_len] for col in row])
    return augmented

def clip(x,n):
    return n-1 if x >= n else 0 if x < 0 else x 

def slidding_window(data, args):
    window_size = args.window_size
    stride = args.window_size
    
    columns = args.cate_col + args.cont_col + args.temp_col + ['answerCode']
    l_idx = columns.index('last_problem')
    
    augmented_datas = []
    for row in data:
        seq_len = len(row[0])
        nx = torch.Tensor(row[l_idx]).numpy()
        last_problems = np.where(nx==max(nx))[0]
        n_test = len(last_problems)
        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if n_test <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((n_test - window_size) // stride) + 1

            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                start = last_problems[clip(window_i*stride,n_test)]
                end = last_problems[clip(window_i*stride+window_size,n_test)]
                if end - start > args.max_seq_len :
                    print("window size is bigger than max seqeunce length!")
                for col in row:
                    window_data.append(col[start:end+1])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if not args.sep_grade and args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if n_test != total_len:
                window_data = []
                for col in row:
                    start = last_problems[-window_size]
                    window_data.append(col[start:])
                augmented_datas.append(tuple(window_data))
    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas

def kfold(df, k):
    MostGrade = df[['userID','grade']].groupby('userID').agg({'grade':lambda x : Counter(x.values).most_common()[0]})

    MostGrade['count'] = MostGrade['grade'].apply(lambda x : x[1])
    MostGrade['grade'] = MostGrade['grade'].apply(lambda x : x[0])
    sorted_grade = MostGrade.sort_values(by = ['grade','count'], axis = 0, ascending=False).reset_index()
    sorted_grade['fold'] = [i%k for i in range(sorted_grade.shape[0])]
    folds = []
    for i in range(k):
        i_index = sorted_grade[sorted_grade['fold']==i]
        folds.append(i_index['userID'].unique())
    return folds

def ELO_function (df) :
    def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return theta + learning_rate_theta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )
    def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return beta - learning_rate_beta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )
    def learning_rate_theta(nb_answers):
        return max(0.3 / (1 + 0.01 * nb_answers), 0.04)
    def learning_rate_beta(nb_answers):
        return 1 / (1 + 0.05 * nb_answers)
    def probability_of_good_answer(theta, beta, left_asymptote):
        return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def estimate_parameters(answers_df, granularity_feature_name='assessmentItemID'):
        item_parameters = {
            granularity_feature_value: {"beta": 0, "nb_answers": 0}
            for granularity_feature_value in np.unique(answers_df[granularity_feature_name])
        }
        student_parameters = {
            student_id: {"theta": 0, "nb_answers": 0}
            for student_id in np.unique(answers_df.userID)
        }
        print("Parameter estimation is starting...")
        for student_id, item_id, left_asymptote, answered_correctly in tqdm(
            zip(answers_df.userID.values, answers_df[granularity_feature_name].values, answers_df.left_asymptote.values, answers_df.answerCode.values)
        ):
            theta = student_parameters[student_id]["theta"]
            beta = item_parameters[item_id]["beta"]
            item_parameters[item_id]["beta"] = get_new_beta(
                answered_correctly, beta, left_asymptote, theta, item_parameters[item_id]["nb_answers"],
            )
            student_parameters[student_id]["theta"] = get_new_theta(
                answered_correctly, beta, left_asymptote, theta, student_parameters[student_id]["nb_answers"],
            )
            item_parameters[item_id]["nb_answers"] += 1
            student_parameters[student_id]["nb_answers"] += 1
        print(f"Theta & beta estimations on {granularity_feature_name} are completed.")
        return student_parameters, item_parameters
    def gou_func (theta, beta) :
        return 1 / (1 + np.exp(-(theta - beta)))
    df['left_asymptote'] = 0
    print(f"Dataset of shape {df.shape}")
    print(f"Columns are {list(df.columns)}")
    student_parameters, item_parameters = estimate_parameters(df)
    prob = [gou_func(student_parameters[student]['theta'], item_parameters[item]['beta']) for student, item in zip(df.userID.values, df.assessmentItemID.values)]
    df['elo_prob'] = prob
    return df