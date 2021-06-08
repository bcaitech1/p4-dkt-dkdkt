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
    # test_model = ["model_04t222du.pt","model_wny713zy.pt","model_ol1he6q5.pt"]
    # aug = [True,False,False]
    # cv = [True,True,True]
    # it = ['problem_number','problem_number','problem_number']
    # lt = ["bce","bce","bce"]
    # sep = [False,True,False]
    # for a1, a2, a3, a4, a5, a6 in zip(aug, cv, it, lt, sep, test_model) :
    #     args.output_name = f"output_{a6[:-3]}.csv"
    #     args.model_name = a6
    #     args.augment = a1
    #     args.cv_strategy = a2 
    #     args.interaction_type = a3
    #     args.loss_type = a4 
    #     args.sep_grade = a5     
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)