import os
from args import parse_args
from dkt.dataloader import Preprocess, add_features, post_process, sweep_apply
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
    train_data, valid_data = preprocess.split_data(train_data, ratio=0.8)

    train_data = post_process(train_data, args)
    valid_data = post_process(valid_data, args)

    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    wandb.login()
    # wandb.init(project='dkt')
    args = parse_args(mode='train')
    # args = sweep_apply(args, wandb.config)
    # args.model_name = f"model_{wandb.run.id}.pt"
    # print(f"start the session {args.model_name}")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)