import os
from args import parse_args
from dkt.dataloader import *
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
    wandb.config.update(args)
    # wandb.init(entity = 'dkdkt', project='DL_model',config= args)
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    TRAIN = preprocess.get_train_data()
    predictions = []
    oof = []
    
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name, is_train_=False)
    TEST_DATA = preprocess.get_test_data()
    TEST_DATA = post_process(TEST_DATA, args)
    preprocess.load_test_data(args.test_file_name, is_train_=True)
    test_data = preprocess.get_test_data()
    if args.kfold > 0 :
        kf = kfold(TRAIN,k=args.kfold)
        for i, valid_idx in enumerate(kf):
            train_data = TRAIN[~TRAIN['userID'].isin(valid_idx)]
            valid_data = TRAIN[TRAIN['userID'].isin(valid_idx)]         
            if args.test2train :
                train_data = pd.concat([train_data, test_data[test_data['answerCode']!=-1]])
            
            print(f'train_data : {train_data.shape[0]}')
            print(f'valid_data : {valid_data.shape[0]}')
            print(f'train_data users: {len(train_data["userID"].unique())}')
            print(f'valid_data users: {len(valid_data["userID"].unique())}')
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
            args.model_name = f"model_{i}.pt"
            args.output_name = f"output_{i}.csv"
            VALID = trainer.run(args, train_data, valid_data)
            predictions.append(VALID)
            trainer.inference(args, TEST_DATA)
            
    # with open(os.path.join(args.model_dir,'folds.pkl'),'wb') as file:
    #     pickle.dump(folds, file)
    with open(os.path.join(args.model_dir,'oof.pkl'),'wb') as file:
        pickle.dump(predictions, file)
    with open(os.path.join(args.model_dir,'arg.pkl'),'wb') as file:
        pickle.dump(args, file)

if __name__ == "__main__":
    wandb.login()
    wandb.init(entity = 'dkdkt', project='DL_model')
    args = parse_args(mode='train')
    # args = sweep_apply(args, wandb.config)
    args.model_name = f"model_{wandb.run.name}.pt"
    args.output_name = f"output_{wandb.run.name}.csv"
    print(f"start the session {args.model_name}")
    args.model_dir = os.path.join(args.model_dir,wandb.run.name)
    args.output_dir = args.model_dir
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)