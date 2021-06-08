from dkt.utils import preprocess_inf_arg
import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
def main(args):
    device = "cuda" if torch.cuda.is_available() and args.device =="gpu" else "cpu"
    args.device = device

    args = preprocess_inf_arg(args)
    print("inference args : \n",args)
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_data)
    test_data = preprocess.get_test_data()
    

    trainer.inference(args, test_data)
    

if __name__ == "__main__":
    args = parse_args(mode='inference')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)