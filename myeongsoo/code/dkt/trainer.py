import os
import torch
import numpy as np


from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion, epoch_update_gamma
from .metric import get_metric
from .model import LSTM, Bert, LSTMATTN, LastQuery

import wandb

def run(args, train_data, valid_data):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10
    args.epoch = 0

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)
        
        ### VALID
        auc, acc,_ , _ = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                    "valid_auc":auc, "valid_acc":acc})
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                },
                args.model_dir, args.model_name,
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
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
        # TODO 8 : 변경한 batch에 따라 3숫자 바꾸기
        targets = input['answerCode'] # correct


        loss = compute_loss(preds, targets, args)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
        
        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
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

    if args.loss_type == 'roc_star':
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
        targets = input['answerCode'] # correct


        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]
    
        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
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
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    write_path = os.path.join(args.output_dir, args.output_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'lstmattn': model = LSTMATTN(args)
    if args.model == 'bert': model = Bert(args)
    if args.model == 'lastquery': model = LastQuery(args)

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):
    # TODO 7 : 변경한 batch에 따라 3숫자 바꾸기
    columns = args.cate_col + args.cont_col + args.temp_col + ['answerCode', "mask"] # ['answerCode', "mask", "interaction", "problem_interaction", "gather_index"]
    batch_dict = dict(zip(columns,batch))
    pr_batch_dict = dict(zip(columns,[0 for _ in columns]))
    # change to float

    pr_batch_dict["mask"] = batch_dict['mask'].type(torch.FloatTensor)
    pr_batch_dict["answerCode"] = batch_dict['answerCode'].type(torch.FloatTensor)
    pr_batch_dict["problem_number"] = batch_dict['problem_number'].type(torch.FloatTensor)
    
    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction_mask = pr_batch_dict["mask"].roll(shifts=1,dims=1)
    interaction_mask[:,0] = 0

    if args.interaction_type == 'base':
        pr_batch_dict["interaction"] = pr_batch_dict["answerCode"] + 1 # 패딩을 위해 correct값에 1을 더해준다.
        pr_batch_dict["interaction"] = pr_batch_dict["interaction"].roll(shifts=1, dims=1)
        pr_batch_dict["interaction"] = (pr_batch_dict["interaction"] * interaction_mask).to(torch.int64)
    
    if args.interaction_type == 'problem_number':
        pr_batch_dict["problem_interaction"] = pr_batch_dict["answerCode"] + 2 * pr_batch_dict["problem_number"] + 1 # 패딩을 위해 correct값에 1을 더해준다.
        pr_batch_dict["problem_interaction"] = pr_batch_dict["problem_interaction"].roll(shifts=1, dims=1)
        pr_batch_dict["problem_interaction"] = (pr_batch_dict["problem_interaction"] * interaction_mask).to(torch.int64)
    # print(interaction)
    # exit()
    
    #  test_id, question_id, tag
    for c in args.cate_col :
        pr_batch_dict[c] = ((batch_dict[c] + 1) * pr_batch_dict["mask"]).to(torch.int64)
    for c in args.cont_col :
        pr_batch_dict[c] = ((batch_dict[c]) * pr_batch_dict["mask"]).to(torch.float32)
        
    pr_batch_dict["answerCode"] = batch_dict['answerCode'].type(torch.LongTensor)
    # gather index
    # 마지막 sequence만 사용하기 위한 index
    pr_batch_dict["gather_index"] = torch.tensor(np.count_nonzero(pr_batch_dict["mask"], axis=1))
    pr_batch_dict["gather_index"] = pr_batch_dict["gather_index"].view(-1, 1) - 1

    # continuous
    args.temp_col
    for c in list(pr_batch_dict.keys()) :
        if c in args.temp_col :
            del(pr_batch_dict[c])
        else :
            pr_batch_dict[c] = pr_batch_dict[c].to(args.device)
    # device memory로 이동
    return pr_batch_dict


# loss계산하고 parameter update!
def compute_loss(preds, targets, args):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets, args)
    #마지막 시퀀드에 대한 값만 loss 계산
    if args.loss_type == 'bce' or args.epoch == 0:
        loss = loss[:,-1]
        loss = torch.mean(loss)

    return loss


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()



def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...', os.path.join(model_dir, model_filename))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))



def load_model(args):
    
    
    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
    
    print("Loading Model from:", model_path, "...Finished.")
    return model