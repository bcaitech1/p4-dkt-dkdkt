import os
import torch
import numpy as np
from tqdm.auto import tqdm

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import epoch_update_gamma, get_criterion
from .metric import get_metric
from .model import LSTM, LSTMATTN, LGBM, LastQuery, TfixupSaint
from dkt.utils import duplicate_name_changer, tensor_dict_to_str
import json
import gc
import wandb

def run(args, train_data, valid_data):
    # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
    torch.cuda.empty_cache()
    gc.collect()

    # augmentation
    augmented_train_data = data_augmentation(train_data, args)
    if len(augmented_train_data) != len(train_data):
        print(f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")

    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) /
                           args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10
    args.epoch = 0

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    log_json = {}
    if args.model_alias != '':
        folder_name = args.model_alias
        print("using model alias")
    else: 
        folder_name = args.model
    model_name = duplicate_name_changer(
        args.model_dir, f"{folder_name}{args.save_suffix}")
    # save_dir = os.path.join(
    #     f"{args.model_dir}{model_name}", str(args.k_fold_idx))
    save_dir = os.path.join(args.model_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    for epoch in tqdm(range(args.n_epochs)):
        
        print(f"Start Training: Epoch {epoch + 1}")

        # TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, args)

        # VALID
        auc, acc, _, _ = validate(valid_loader, model, args)

        result = {"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc,
                "valid_auc": auc, "valid_acc": acc}
        # TODO: model save or early stopping
        wandb.log(result)

        result["train_loss"] = train_loss.item()
        log_json[epoch] = tensor_dict_to_str(result)

        with open(f'{save_dir}/log.json', 'w') as f:
            json.dump(log_json, f, indent=4)

        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
            },
                tensor_dict_to_str(result),
                save_dir, 'best.pt', args
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()


def train(train_loader, model, optimizer, args):
    model.train()
    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)
        preds = model(input)
        targets = input['oth']['answerCode']  # answerCode

        loss = compute_loss(preds, targets, args)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            # tensor를 numpy화할 때, gpu에서 진행불가, cpu로 넘겨야 함.
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc}')

    if args.loss == 'roc_star' or args.loss == 'both':
        total_targets, total_preds = torch.Tensor(total_targets).to(args.device), torch.Tensor(total_preds).to(args.device)
        args.gamma = epoch_update_gamma(total_targets, total_preds, epoch=args.epoch, delta = args.delta)
        args.last_target = total_targets 
        args.last_predict = total_preds
        args.epoch += 1
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input['oth']['answerCode']  # answerCode

        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets


def inference(args, test_data):

    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
        input = process_batch(batch, args)

        preds = model(input)

        # predictions
        preds = preds[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()

        total_preds += list(preds)
    
    output_name = duplicate_name_changer(
        args.output_dir, f"output{args.save_suffix}.csv")

    write_path = os.path.join(args.output_dir, output_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id, p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm':
        model = LSTM(args)
    elif args.model == 'lstmattn':
        model = LSTMATTN(args)
    elif args.model == 'lstqry':
        model = LastQuery(args)
    elif args.model == 'saint':
        model = TfixupSaint(args)
    elif 'tfixup' in args.model:
        model = TfixupSaint(args)

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):

    # test, question, tag, correct, mask = batch
    batch_dict = {args.column_seq[i]: col for i, col in enumerate(batch)}
    other_dict = {}

    # change to float
    other_dict['mask'] = batch_dict['mask'] = batch_dict['mask'].type(
        torch.FloatTensor)
    other_dict['answerCode'] = batch_dict['answerCode'] = batch_dict['answerCode'].type(
        torch.FloatTensor)
    
    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction = batch_dict['answerCode'] + 1  # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = batch_dict['mask'].roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)
    cont_dict, cate_dict = {}, {}
    for col_name in batch_dict.keys():
        if col_name != "mask" and col_name != "answerCode":
            if col_name in args.cont_cols:
                cont_dict[col_name] = (
                    batch_dict[col_name] * batch_dict['mask']).to(torch.float32).to(args.device)
            else:
                cate_dict[col_name] = (
                    (batch_dict[col_name] + 1) * batch_dict['mask']).to(torch.int64).to(args.device)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(batch_dict['mask'], axis=1))
    gather_index = gather_index.view(-1, 1) - 1

    # device memory로 이동
    # for col_name in batch_dict.keys():
    #     batch_dict[col_name] = batch_dict[col_name].to(args.device)
    other_dict['interaction'] = interaction.to(args.device)
    other_dict['gather_index'] = gather_index.to(args.device)
    other_dict['mask'] =other_dict['mask'].to(args.device)
    other_dict['answerCode'] = other_dict['answerCode'].to(args.device)
    return {
        "cont": cont_dict,
        "cate": cate_dict,
        "oth": other_dict
    }


# loss계산하고 parameter update!
def compute_loss(preds, targets, args):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets, args)
    # 마지막 시퀀드에 대한 값만 loss 계산    
    if args.loss == 'bce' or args.epoch == 0:
        loss = loss[:, -1]
        loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, result, save_dir, model_filename, args):
    print('saving model ...')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, model_filename))
    with open(save_dir+'/best.json', 'w') as f:
        json.dump(result, f, indent=4)

    wandb_config = ['model','loss','val_data','max_seq_len', 'lr', 'drop_out', 'hidden_dim', 'n_layers', 'batch_size', 'optimizer', 'train_data','test_data', 'n_epochs', 'n_layers', 'n_heads' ]
    config_dict = {k:v for k, v in vars(args).items() if k in wandb_config}

    with open(save_dir+'/model_config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)

def load_model(args):
    if hasattr(args, 'model_path'):
        model_path =  args.model_path
    else:
        model_path = os.path.join(args.model_dir, args.model_name)
        model_path = os.path.join(model_path, "model_config.json")
    if not os.path.exists(model_path): raise FileExistsError(f"dir {model_path} doesn't exists.")
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model


def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.stride

    augmented_datas = []
    for row in tqdm(data):
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1
            
            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))
    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas


def data_augmentation(data, args):
    if args.window == True:
        print("####Data augmentation####")
        data = slidding_window(data, args)

    return data