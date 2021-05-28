import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds, preprocess_arg
import wandb
def main(args):
    wandb.login()

    preprocess_arg(args)
    
    setSeeds(args.seed)

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    
    train_data, valid_data = preprocess.split_data(train_data)
    
    wandb.init(project='dkt', config=vars(args))
    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    print(f"train args : \n {args}")      
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)