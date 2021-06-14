import os
import argparse
import json

def parse_args(mode='train'):
    parser = argparse.ArgumentParser()
    # custom Feature engineerings
    parser.add_argument('--fe_dir', default="./features/", type=str, help='directory of FE set. (default: ./features/)')
    parser.add_argument('--fe_set', default="MF_FE.json", type=str, help='name of FE Set. (default: default_FE.json)')
   
    # custom config input output
    parser.add_argument('--json', nargs='?', const='latest', type=str, help='get argument form json file. (default: get lastets file from config/train)' )
    parser.add_argument('--exp_cfg', nargs='?', const='./config/train/export/exported_config.json', type=str, help='Directory and name of exported config.')
    parser.add_argument('--no_select', nargs='?', const=True, type=bool, help='Select config json from directory given at "json" argument. (default: False)' )

    parser.add_argument('--seed', default=42, type=int, help='seed')
    
    parser.add_argument('--device', default='gpu', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='/opt/ml/input/data/train_dataset', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='asset/', type=str, help='data directory')
    
    parser.add_argument('--train_data', default='cv_train_data_FE_MF5.pkl', type=str, help='train file name')
    parser.add_argument('--val_data', nargs='?', const='cv_valid_data_FE_MF5.pkl', type=str, help='validation file name')
    parser.add_argument('--test_data', default='cv_test_data_FE_MF5.pkl', type=str, help='test file name')

    parser.add_argument('--datasets', nargs='?', const='{"train":"cv_train_data_FE_MF5.pkl", "valid":"cv_valid_data_FE_MF5.pkl","test":"cv_test_data_FE_MF5.pkl"}', type=json.loads, help='dataset set')

    parser.add_argument('--model_dir', default='models/', type=str, help='model directory(default: models/)')
    parser.add_argument('--model_alias', default='', type=str, help='model output folder name(default: {your model name}/)')    
    parser.add_argument('--save_suffix', default='', type=str, help='suffix for saving file(default: None)')

    parser.add_argument('--output_dir', default='output/', type=str, help='output directory')
    parser.add_argument('--inf_config', type=str, default="lstm/",help='model file name like "{model_folder}/" or "{model_folder}" (default: lstm/))')
    
    parser.add_argument('--max_seq_len', default=500, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--pin_mem', default=True, type=bool, help='using pin memory in dataloader(default:True)')
    
    # 모델
    parser.add_argument('--hidden_dim', default=128, type=int, help='hidden dimension size')
    parser.add_argument('--n_layers', default=4, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=4, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.27, type=float, help='drop out rate')
    
    # 훈련
    parser.add_argument('--n_epochs', default=30, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.00007, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=100, type=int, help='clip grad')
    parser.add_argument('--patience', default=10, type=int, help='for early stopping')
    
    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')
    
    ### 중요 ###
    parser.add_argument('--model', default='lstmattn', type=str, help='model type')
    parser.add_argument('--optimizer', default='adamW', type=str, help='optimizer type')
    parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler type')
    parser.add_argument('--loss', default='bce', type=str, help='loss type')
    parser.add_argument('--delta', default=2, type=float, help='loss gamma delta')
    
    args = parser.parse_args()

    return args