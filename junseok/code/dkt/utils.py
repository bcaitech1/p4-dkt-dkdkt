import os
import random
import torch
import numpy as np
import argparse
import json
import glob
import enquiries
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

def setSeeds(seed=42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def convert_time(s):
    timestamp = time.mktime(datetime.strptime(
        s, '%Y-%m-%d %H:%M:%S').timetuple())
    return int(timestamp)


def get_lgbm_dataset(args):  
    ordinal_feats = ['grade']
    label_feats = ['assessmentItemID', 'problem_number','hour','dow','solved_disorder','KnowledgeTag','testId','retest']
    datas = []
    for file_name in [args.train_data, args.val_data, args.test_data]:
        csv_file_path = os.path.join(args.data_dir, file_name)
        if '.pkl' in file_name:
            print("loading file from pkl")
            df = pd.read_pickle(csv_file_path)
        else:
            df = pd.read_csv(csv_file_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            print("drop Unnamed: 0 column")
        datas.append(df)
        
    for col in datas[0].columns:
            if col in label_feats:
                for idx, data in enumerate(datas):
                    X = data[col].values.reshape(-1,1)
                    if idx == 0:
                        enc = LabelEncoder()
                        enc.fit(X)  
                    X = enc.transform(X)
                    data[col] = X
            elif col in ordinal_feats:                
                for idx, data in enumerate(datas):
                    X = data[col].values.reshape(-1,1)
                    if idx == 0:
                        enc = OrdinalEncoder()
                        enc.fit(X)
                    X = enc.transform(X)
                    data[col] = X
    return datas


def get_latest_created_file(data_dir="./config/train/", file_type="json") -> str:
    # Get latest file from given directory default: json
    print(f"get latest created {file_type} file from {data_dir} ...")
    list_of_files = glob.glob(f'{data_dir}*.{file_type}')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"use {latest_file} file")
    return latest_file


def get_col_type(df: pd.DataFrame):
    cate_types = ['str', 'string', 'object', 'category']
    column_mask = ['userID', 'Timestamp', 'answerCode']
    cate_cols, cont_cols = [], []
    for column_name in df.columns:
        if column_name in column_mask:
            continue
        if str(df[column_name].dtype) in cate_types:
            cate_cols.append(column_name)
        else:
            cont_cols.append(column_name)

    return cate_cols, cont_cols


def hidden_dim_allocator(input_dim, cols, mode="uniform"):
    if mode == "uniform":
        pass
    elif mode == "halven":
        pass
    elif mode == "size":
        pass


def get_latest_modified_file(data_dir="./config/train/", file_type="json") -> str:
    # Get latest file from given directory default: json
    print(f"get latest modified {file_type} file from {data_dir} ...")
    list_of_files = glob.glob(f'{data_dir}*.{file_type}')
    latest_file = max(list_of_files, key=os.path.getmtime)
    print(f"use {latest_file} file")
    return latest_file


def dislplay_file_dir(file_list):
    # Display file selection list.
    if not file_list:
        file_list = ['↰ to the parent directory (..) \n Nothing here... 🙊']
    else:
        file_list = list(map(lambda x: '┣ '+x, file_list))
        file_list[-1] = '┗'+file_list[-1][1:]
        file_list = ['↰ to the parent directory (..)'] + file_list
    return file_list


def select_file_from_dir(init_dir, target_type=None):
    # Select file and Get file path, *(It only works at Linux, MAC system).
    init_dir = os.path.abspath(init_dir)
    while True:
        list_of_files = glob.glob(f'{init_dir}/*')
        file_list = sorted(list_of_files, key=os.path.getmtime)
        processed_file_list = dislplay_file_dir(file_list)
        selected = enquiries.choose(
            f'Select {target_type} file from {init_dir} (ctrl+c to exit): \n', processed_file_list)
        if os.path.isfile(selected[2:]) and (not target_type or f'.{target_type}' in selected):
            # proper file seleted, returned it.
            return selected[2:]
        elif selected == processed_file_list[0]:
            # get to the parent directoy.
            init_dir = Path(init_dir).parent.absolute()
        elif os.path.isdir(selected[2:]):
            # folder selected, change directory.
            init_dir = selected[2:]
        elif os.path.isfile(selected[2:]) and f'.{target_type}' not in selected:
            # wrong type file selected, warn and reselet.
            print("***************************************************")
            print(f"**Wrong file type detected. select .{target_type} file.**")
            print("***************************************************")


def check_wandb_json(config) -> bool:
    # Check if json file from wandb.ai.
    return list(config.values())[0] is dict and 'desc' in list(config.values())[0].keys()


def import_data_from_json(json_file: str, return_type="json"):
    # Import config(json form) from directory.
    with open(json_file) as jf:
        data = json.load(jf)

    # Fix inappropriate structure.
    if return_type == "json" and check_wandb_json(data):
        print("convert wandb json to normal json")
        for k, v in data.items():
            data[k] = v['value']
    if return_type == "argparse":
        data = argparse.Namespace(**data)
    return data


def duplicate_name_changer(target_dir, fname):
    # Prevent duplicate folder name by adding suffix '_{n}'.
    idx = 1
    while os.path.exists(target_dir+fname):
        fname_sep = fname.split('.')
        if idx > 1:
            fname_sep[0] = fname_sep[0][:-(len(str(idx-1))+1)]
        fname_sep[0] += f'_{idx}'
        fname = '.'.join(fname_sep)
        idx += 1
    return fname


def tensor_dict_to_str(target_dict):
    for k, i in target_dict.items():
        target_dict[k] = str(i)
    return target_dict


def export_config_as_json(config,  input_dir: str):
    # Export json.

    # get directory separator form given input_dir.
    if '/' in input_dir:
        sep = '/'
    elif '\\' in input_dir:
        sep = '\\'
    else:
        raise RuntimeError('invalid directory')
    # split by separator and get file_name and directory.
    input_dir = input_dir.split(sep)
    file_name = input_dir[-1]
    export_dir = '/'.join(input_dir[:-1]) + '/'
    os.makedirs(export_dir, exist_ok=True)
    file_name = duplicate_name_changer(export_dir, file_name)

    with open(export_dir+file_name, 'w') as outfile:
        json.dump(vars(config), outfile, indent=4)


def get_batch_size(config: dict):
    # config values 중, list type이고, batch_size가 모두 같아야 하며, 해당 batch_size return
    config_vals = [v for _, v in config.items() if type(v) == list]
    batch_size = len(config_vals[0]) if config_vals else None
    if any(len(i) != batch_size for i in config_vals):
        raise RuntimeError(
            f"some length of argument doesn't match with other batched arguments. check your json file.")
    return batch_size


def batch_json_processing(config: argparse.Namespace):

    wand_db_argu = ['n_tag', '_wandb', 'n_test']
    store_argu = ['column', 'non_cate_col', 'cate_col']
    input_argu = ["json", "exp_cfg", "no_select"]
    unavailable = (wand_db_argu+store_argu+input_argu)
    config = vars(config)
    arg_list = {}
    batch_size = get_batch_size(config)
    if batch_size:
        result = []
        for arg_name, arg in config.items():
            if type(arg) == list:
                arg_list[arg_name] = arg
            else:
                arg_list[arg_name] = [arg for _ in range(batch_size)]
        for i in range(batch_size):
            arg_ele = {k: None for k in config.keys() if k not in unavailable}
            for k in arg_ele.keys():
                arg_ele[k] = arg_list[k][i]
            arg_ele['save_suffix'] += "_batched"
            result.append(argparse.Namespace(**arg_ele))
        return result
    else:
        return [argparse.Namespace(**config)]


def preprocess_arg(args: argparse.Namespace):
    # preprecess arguments for usage.
    # Export config (*Must be first job).
    if hasattr(args, 'exp_cfg') and args.exp_cfg != None:
        print("config exporting...")
        export_config_as_json(args, args.exp_cfg)

    # Get config from json:
    if not args.no_select:
        # for json select mode
        if not hasattr(args, 'json') or args.json != 'lastest':
            target_dir = './config/train/'
        else:
            target_dir = args.json
        os.makedirs(target_dir, exist_ok=True)
        selected = select_file_from_dir(target_dir, 'json')
        args = import_data_from_json(selected, 'argparse')
    elif hasattr(args, 'json') and args.json != None:
        # for json directory mode
        if args.json == "latest":
            args.json = get_latest_modified_file()
        args = import_data_from_json(args.json, 'argparse')
    args_list = []
    for args in batch_json_processing(args):
        if hasattr(args, 'datasets') and args.datasets != None:
            args.train_data = args.datasets['train']            
            args.val_data = args.datasets['valid']         
            args.test_data = args.datasets['test']

        # Change device based on server system.
        if torch.cuda.is_available() and (args.device == "gpu" or args.device == "cuda"):
            device = "cuda"
        else:
            if not torch.cuda.is_available() and (args.device == "gpu" or args.device == "cuda"):
                print(
                    '*'*10, "CUDA Unavailable! Automatically change device to CPU", '*'*10)
            device = "cpu"
        args.device = device
        args_list.append(args)
    return args_list


def preprocess_inf_arg(args: argparse.Namespace):
    # preprecess inference arguments for usage.

    # Get model from select mode:
    if args.no_select:
        # if args.model_name != '.json':
        #     raise ImportError("wrong model_name argument input.")
        mini_args = import_data_from_json(os.path.join(args.model_dir, args.inf_config))
    else:
        selected = select_file_from_dir('./models/', 'json')
        if "model_config.json" not in selected:
            raise ImportError("Wrong input, Select model_config.json.")
        mini_args = import_data_from_json(selected)
    args = vars(args)
    for k, v in mini_args.items():
        args[k] = v

    return argparse.Namespace(**args)
