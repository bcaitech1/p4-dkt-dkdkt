import os
from args import parse_args
from dkt.dataloader import Preprocess, add_features, post_process, sweep_apply, slide_window
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
import numpy as np


def main(args):
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    args = add_features(args)
    wandb.init(project='dkt',config= args)
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    if args.cv_test :
        print("use a testset as a valid set")
        preprocess.load_test_data(args.test_file_name)
        valid_data = preprocess.get_test_data()
    else :
        train_data, valid_data = preprocess.split_data(train_data, ratio=0.8)
        if args.test2train :
            preprocess.load_test_data(args.test_file_name)
            test_data = preprocess.get_test_data()
            train_data = pd.concat([train_data, test_data])
        
    print(f'train_data : {train_data.shape[0]}')
    print(f'valid_data : {valid_data.shape[0]}')
    print(f'train_data users: {len(train_data["userID"].unique())}')
    print(f'valid_data users: {len(valid_data["userID"].unique())}')

    # train_data = pd.concat([train_data, test_data[test_data['answerCode']!=-1]])

    train_data = post_process(train_data, args)
    valid_data = post_process(valid_data, args)
    
    print(f'after post processing train_data : {len(train_data)}')
    print(f'after post processing valid_data : {len(valid_data)}')

    
    if args.slide : 
        train_data = slide_window(train_data,args.hm_tr,'train', args)
        print(f'after sliding window train_data : {len(train_data)}')    
    if args.cv_test :
        valid_data = slide_window(valid_data,args.hm_ts,'test', args)
        print(f'after sliding window valid_data : {len(valid_data)}')

    trainer.run(args, train_data, valid_data)
    
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    test_data = post_process(test_data, args)
    trainer.inference(args, test_data)

if __name__ == "__main__":
    wandb.login()
    wandb.init(project='dkt')
    args = parse_args(mode='train')
    args = sweep_apply(args, wandb.config)
    args.model_name = f"model_{wandb.run.name}.pt"
    args.output_name = f"output_{wandb.run.name}.csv"
    print(f"start the session {args.model_name}")
    os.makedirs(args.model_dir, exist_ok=True)
    if args.cv_test==False and (args.hm_tr > 2 or args.hm_ts > 2):
        print('dont repeat same condition')
        pass
    elif args.slide :
        pass
    elif args.cv_test and args.test2train :
        print('controversy!')
        pass
    else :
        main(args)