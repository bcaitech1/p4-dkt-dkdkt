import os
from args import parse_args
from dkt.dataloader import Preprocess, add_features, post_process
from dkt import trainer
import torch
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    args = add_features(args)
    
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    test_data = post_process(test_data, args)
    trainer.inference(args, test_data)

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)