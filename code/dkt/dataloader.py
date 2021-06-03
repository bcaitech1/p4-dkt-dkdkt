from genericpath import exists
import os
from typing import Dict
from pandas.core.frame import DataFrame
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from importlib import import_module
from tqdm.auto import tqdm
from code.dkt.utils import convert_time, get_col_type, import_data_from_json


class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.valid_data = None
        

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data
        
    def get_valid_data(self):
        return self.valid_data

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

        os.makedirs(self.args.asset_dir,exist_ok=True)

        for col in tqdm(self.args.cate_cols, desc="preprocessing categorical data..."):            
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
         
        if df['Timestamp'].dtype == object:
            print("Processing Timestamp...")
            df['Timestamp'] = df['Timestamp'].apply(convert_time)   
            print("Processing Timestamp done")
 

        return df

    def __feature_engineering(self, df):        
        cate_types = ['str', 'string', 'object', 'category']
        # cont_types = ['int', 'float', 'long', 'int64', 'float64', 'int32', 'datetime', 'datetime64']
        fe_path = os.path.join(self.args.fe_dir, self.args.fe_set)
        fe_set = import_data_from_json(fe_path)
        print(f"Using {self.args.fe_set} file for Featre Engineering.")
        if fe_set['base_dataset'] != self.args.file_name: print("Warning! Input dataset is not matched with FE target dataset.")
        self.args.cate_cols, self.args.cont_cols = get_col_type(df)
        for col_name, col_info in fe_set['featrues'].items():
            if col_info != None and col_info != "del": 
                df[col_name] = pd.Series.from_csv(col_info['column'])
                df[col_name].astype(col_info['type'].lower())
                if col_info['type'].lower() in cate_types: self.args.cate_cols.append(col_name)
                else: self.args.cont_cols.append('col_info')
            else:
                df = df.drop(col_name)
                if col_name in self.args.cate_cols:
                    self.args.cate_cols.remove(col_name)
                elif col_name in self.args.cont_cols:
                    self.args.cont_cols.remove(col_name)
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, nrows=1000)
        df = self.__feature_engineering(df)
        df, cate_cols = self.__preprocessing(df, is_train)
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        cate_cols = [i for i in self.args.cate_cols]
        self.args.cate_cols = {}
        for col in cate_cols:
            self.args.cate_cols[col] = (len(np.load(os.path.join(self.args.asset_dir,f'{col}_classes.npy'))))

        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        
        group = df[cate_cols + self.args.cont_cols].groupby('userID').apply(set_column)

        self.args.column_seq = group.columns
        self.args.column_seq.remove('userID')
        self.args.column_seq.remove('Timestamp')
        self.args.column_seq.append('mask')
        
        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)

    def load_valid_data(self, file_name):
        self.valid_data = self.load_data_from_file(file_name, is_train= False)

def set_column(r):
    new_df = []
    columns = list(r.columns)
    if 'Timestamp' in columns: columns.remove('Timestamp')
    if 'userID' in columns: columns.remove('userID')
    for column in columns:
        new_df.append(r[column].values)
    return new_df

class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])        
        
        # test, question, tag, correct = row[0], row[1], row[2], row[3]
        
        # need to refactoring!
        cols = [row[i] for i, col_name in enumerate(self.args.column_seq) if col_name != 'mask']

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

    pin_memory = args.pin_mem
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