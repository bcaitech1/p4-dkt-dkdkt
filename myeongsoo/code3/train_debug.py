import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from args import parse_args
from dkt.dataloader import slidding_window, Preprocess, add_features, post_process, sweep_apply, slide_window
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
import numpy as np
import pandas as pd

def main(args):
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    args = add_features(args)
    wandb.init(entity = 'dkdkt', project='DL_model',config= args)
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    
    if args.cv_test :
        print("use a testset as a valid set")
        preprocess.load_test_data(args.test_file_name)
        valid_data = preprocess.get_test_data()
    else :
        train_data, valid_data = preprocess.split_data(train_data, ratio=0.8)
        
    print(f'train_data : {train_data.shape[0]}')
    print(f'valid_data : {valid_data.shape[0]}')
    print(f'train_data users: {len(train_data["userID"].unique())}')
    print(f'valid_data users: {len(valid_data["userID"].unique())}')

    # train_data = pd.concat([train_data, test_data[test_data['answerCode']!=-1]])
    # train_data = train_data[:20000]
    # valid_data = valid_data[:10000]
    train_data = post_process(train_data, args)
    valid_data = post_process(valid_data, args)
    
    print(f'after post processing train_data : {len(train_data)}')
    print(f'after post processing valid_data : {len(valid_data)}')

    
    if args.slide : 
        train_data = slidding_window(train_data, args)
        # train_data = slide_window(train_data,args.hm_tr,'train', args)
        print(f'after sliding window train_data : {len(train_data)}')    
    if args.cv_test :
        valid_data = slide_window(valid_data,args.hm_ts,'test', args)
        print(f'after sliding window valid_data : {len(valid_data)}')

    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    wandb.login()
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)