import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
from dkt.utils import setSeeds, preprocess_arg, get_lgbm_dataset
from dkt.model import LGBM
import wandb

wandb_config = ['model','loss','val_data','max_seq_len', 'lr', 'drop_out', 'hidden_dim', 'n_layers', 'batch_size', 'optimizer' ]

def main(args):
    wandb.login()

    args_list = preprocess_arg(args)

    wandb.init(project='dkt', config=vars(args), name=str({k:v for k,v in vars(args).items() if k in wandb_config}))
    for i, args in enumerate(args_list):
        print(f"train args : \n {args}")
        print(f'start {i} json config')
        args.k_fold_idx = 0
        setSeeds(args.seed)
        if args.model == "lgbm":
            model = LGBM(args)
            train, valid, test = get_lgbm_dataset(args)
            model.train(train,valid, test)
            continue
        preprocess = Preprocess(args)
        preprocess.load_train_data(args.train_data)
        train_data = preprocess.get_train_data()

        if args.val_data:
            print("using validation_dataset...")
            preprocess.load_valid_data(args.val_data)
            valid_data = preprocess.get_valid_data()
        else:
            train_data, valid_data = preprocess.split_data(train_data)

        # wandb.init(project='dkt', config={k:v for k,v in vars(args).items() if k in wandb_config})
        trainer.run(args, train_data, valid_data)

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)