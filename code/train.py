import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds, preprocess_arg
import wandb

def main(args):
    wandb.login()

    args_list = preprocess_arg(args)
    for i, args in enumerate(args_list):        
        print(f"train args : \n {args}")      
        print(f'start {i} json config')
        args.k_fold_idx = 0
        setSeeds(args.seed)

        preprocess = Preprocess(args)
        preprocess.load_train_data(args.file_name)
        train_data = preprocess.get_train_data()

        if args.val_name:
            print("using validation_dataset...")
            preprocess.load_valid_data(args.val_name)
            valid_data = preprocess.get_valid_data()
        else:
            train_data, valid_data = preprocess.split_data(train_data)
    
        wandb.init(project='dkt', config=vars(args))
        trainer.run(args, train_data, valid_data)    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)