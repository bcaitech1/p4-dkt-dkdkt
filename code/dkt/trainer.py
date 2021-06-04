import os
import torch
import numpy as np
from tqdm.auto import tqdm

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM
from dkt.utils import duplicate_name_changer, tensor_dict_to_str
import json

import wandb


def run(args, train_data, valid_data):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) /
                           args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    log_json = {}

    model_name = duplicate_name_changer(
        args.model_dir, f"{args.model}{args.save_suffix}")
    save_dir = os.path.join(
        f"{args.model_dir}{model_name}", str(args.k_fold_idx))
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
            json.dump(log_json, f)
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
            },
                tensor_dict_to_str(result),
                save_dir, 'best.pt',
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

        loss = compute_loss(preds, targets)
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
        # preds = preds[:,-1]
        preds = preds[-1, :]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()

        total_preds += list(preds)

    write_path = os.path.join(args.output_dir, "output.csv")
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
    if args.model == 'lstmattn':
        model = LSTMATTN(args)
    if args.model == 'bert':
        model = Bert(args)

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
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, result, save_dir, model_filename):
    print('saving model ...')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, model_filename))
    with open(save_dir+'/best.json', 'w') as f:
        json.dump(result, f)


def load_model(args):

    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
