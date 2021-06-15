import os
import argparse


def parse_args(mode='train'):
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--seed', default=42, type=int, help='seed')
    
    parser.add_argument('--device', default='cuda', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='/opt/ml/input/data/train_dataset', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='/opt/ml/asset/', type=str, help='data directory')
    parser.add_argument('--feature_type', default='ver1', type=str, help='feature combination you choose')

    parser.add_argument('--file_name', default='train_data.csv', type=str, help='train file name')
    
    parser.add_argument('--model_dir', default='/opt/ml/models/', type=str, help='model directory')
    parser.add_argument('--model_name', default='model_earthy-sweep-12.pt', type=str, help='model file name')
    parser.add_argument('--output_name', default='output.csv', type=str, help='model file name')

    parser.add_argument('--output_dir', default='/opt/ml/output/SAINTSWEEP', type=str, help='output directory')
    parser.add_argument('--test_file_name', default='test_data.csv', type=str, help='test file name')
    
    parser.add_argument('--max_seq_len', default=500, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers')

    # 모델
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dimension size')
    parser.add_argument('--n_layers', default=4, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=2, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.3, type=float, help='drop out rate')
    parser.add_argument('--interaction_type', default='problem_number', type=str, help='how to embed the interaction')
    # 훈련
    parser.add_argument('--n_epochs', default=25, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    parser.add_argument('--patience', default=3, type=int, help='for early stopping')
    parser.add_argument('--sep_grade', default=True, type=bool, help='for seperate the sequence by grade')
    parser.add_argument('--augment_rate', default=0.5, type=float, help='percent to augment')    
    parser.add_argument('--augment', default=True, type=bool, help='for augment the seq length various')
    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')
    parser.add_argument('--cv_strategy', default=True, type=bool, help='for apply cv strategy for train/valid split')
    parser.add_argument('--cv_test', default=False, type=bool, help='for apply cv strategy for train/valid split')
    parser.add_argument('--slide', default=True, type=bool, help='for apply cv strategy for train/valid split')
    parser.add_argument('--window_size', default=10, type=int, help='for apply cv strategy for train/valid split')
    parser.add_argument('--shuffle', default=True, type=bool, help='for apply cv strategy for train/valid split')
    parser.add_argument('--shuffle_n', default=3, type=int, help='for apply cv strategy for train/valid split')
    parser.add_argument('--kfold', default=5, type=int, help='for apply cv strategy for train/valid split')

    parser.add_argument('--cate_col_e_type', default='ver1', type=str, help='for apply cv strategy for train/valid split')
    parser.add_argument('--cate_col_d_type', default='ver2', type=str, help='for apply cv strategy for train/valid split')
    parser.add_argument('--cont_col_e_type', default='ver1_d', type=str, help='for apply cv strategy for train/valid split')
    parser.add_argument('--cont_col_d_type', default='ver2_d', type=str, help='for apply cv strategy for train/valid split')
    
    parser.add_argument('--hm_tr', default=4, type=int, help='for apply cv strategy for train/valid split')
    parser.add_argument('--hs_tr', default=2, type=int, help='for apply cv strategy for train/valid split')
    parser.add_argument('--test2train', default=True, type=bool, help='for apply cv strategy for train/valid split')

    ### 중요 ###
    parser.add_argument('--model', default='gpt', type=str, help='model type')
    parser.add_argument('--optimizer', default='adamW', type=str, help='optimizer type')
    parser.add_argument('--weight_decay', default=0.003, type=float, help='optimizer weight decay')

    parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler type')

    parser.add_argument('--loss_type', default='bce', type=str, help='loss type')
    parser.add_argument('--delta', default=2, type=float, help='scheduler type')
    args = parser.parse_args()

    return args