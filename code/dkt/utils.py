import os, random, torch
import numpy as np
import argparse
import json
import glob
import enquiries
from pathlib import Path

def setSeeds(seed = 42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_latest_created_file(data_dir="./config/train/", file_type="json")->str:    
    # Get latest file from given directory default: json
    print(f"get latest created {file_type} file from {data_dir} ...")
    list_of_files = glob.glob(f'{data_dir}*.{file_type}')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"use {latest_file} file")
    return latest_file


def get_latest_modified_file(data_dir="./config/train/", file_type="json")->str:    
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
        file_list = list(map(lambda x : '┣ '+x,file_list))
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
        selected = enquiries.choose(f'Select {target_type} file from {init_dir} (ctrl+c to exit): \n', processed_file_list)
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
            print("*****Wrong file type detected. select another.*****")
            print("***************************************************")



def check_wandb_json(config)->bool:
    # Check if json file from wandb.ai.
    return list(config.values())[0] and 'desc' in list(config.values())[0].keys()


def import_config_from_json(json_file:str):
    # Import config(json form) from directory.
    with open(json_file) as jf:
        config = json.load(jf)

    # Fix inappropriate structure.
    if check_wandb_json(config):
        print("convert wandb json to json")
        for k,v in config.items():
            config[k] = v['value']

    config = argparse.Namespace(**config) 
    return config


def duplcate_name_changer(target_dir, fname):
    #Prevent duplicate file name by adding suffix '_{n}'.
    idx = 1
    while os.path.exists(target_dir+fname): 
        fname_sep = fname.split('.')
        if idx > 1:
            fname_sep[0] = fname_sep[0][:-2]
        fname_sep[0] += f'_{idx}'
        fname = '.'.join(fname_sep)
        idx += 1
    return fname

def export_config_as_json(config,  input_dir:str):
    #Export json.

    # get directory separator form given input_dir.
    if '/' in input_dir: sep = '/'
    elif '\\' in input_dir: sep = '\\'
    else: raise RuntimeError('invalid directory')

    # split by separator and get file_name and directory.
    input_dir = input_dir.split(sep)
    file_name = input_dir[-1]
    export_dir = '/'.join(input_dir[:-1])
    export_dir += '/'
    os.makedirs(export_dir, exist_ok=True)

    file_name = duplcate_name_changer(export_dir, file_name)
    
    with open(export_dir+file_name,'w') as outfile:
        json.dump(vars(config), outfile)


def preprocess_arg(args:argparse.Namespace):
    # preprecess arguments for usage.

    # Export config (*Must be first job).
    if hasattr(args, 'exp_cfg') and args.exp_cfg != None:
        export_config_as_json(args, args.exp_cfg)

    # Change device based on server system.
    if torch.cuda.is_available() and args.device =="gpu":
        device = "cuda"
    else:
        if not torch.cuda.is_available() and args.device =="gpu":
            print('*'*10,"CUDA Unavailable! Change device to CPU",'*'*10)
        device = "cpu"
    args.device = device 

    # Sort Feature Engeering order.
    if hasattr(args, 'fes'):
        args.fes = sorted(args.fes)
    else:
        print("Warning! Update your code")
        args.fes = []

    # Get config from json:
    if not args.no_select:
        # for json select mode
        target_dir = './config/train/' if not hasattr(args, 'json') or args.json != 'lastest' else args.json 
        os.makedirs(target_dir, exist_ok=True)
        selected = select_file_from_dir(target_dir, 'json')
        args = import_config_from_json(selected) 
    elif hasattr(args, 'json') and args.json != None: 
        # for json directory mode
        if args.json == "latest": 
            args.json = get_latest_modified_file()
        args = import_config_from_json(args.json)  
    return args