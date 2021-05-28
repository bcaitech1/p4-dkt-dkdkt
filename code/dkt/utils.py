import os, random, torch
import numpy as np
import argparse
import json
import glob

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


def export_config_as_json(config,  export_dir:str, fname='default'):
    #Export json.
    os.makedirs(export_dir, exist_ok=True)
    file_name = f'{export_dir}'+fname+'.json'
    with open(file_name,'w') as outfile:
        json.dump(vars(config), outfile)


def preprocess_arg(args:argparse.Namespace):
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
    if args.json: 
        if args.json == "latest": args.json = get_latest_modified_file()
        args = import_config_from_json(args.json)  
    return args