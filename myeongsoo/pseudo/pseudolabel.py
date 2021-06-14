import os
import numpy as np
import pandas as pd
import copy
import torch
import wandb
import gc
from tqdm import tqdm
import pickle

from args import parse_args
from dkt.utils import setSeeds
from dkt.metric import get_metric
from dkt.dataloader import *
from dkt.optimizer import get_optimizer
from dkt.scheduler import get_scheduler
from dkt.criterion import get_criterion
from dkt.trainer import get_model, train, validate, process_batch, save_checkpoint

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Trainer:
    def __init__(self):
        pass

    def train(self, args, train_data, valid_data, model = None):
        """훈련을 마친 모델을 반환한다"""

        # args update
        self.args = args

        # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
        torch.cuda.empty_cache()
        gc.collect()

        # augmentation
        augmented_train_data = slidding_window(train_data, args)
        if len(augmented_train_data) != len(train_data):
            print(f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")

        train_loader, valid_loader = get_loaders(args, augmented_train_data, valid_data)
        
        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
        args.warmup_steps = args.total_steps // 10
            
        model = get_model(args)
        optimizer = get_optimizer(model,None, args)
        scheduler = get_scheduler(optimizer, args)
        early_stopping_counter = 0
        best_auc = -1
        best_model = model # -1
        for epoch in range(args.n_epochs):
            print(f"model training...{epoch}||{early_stopping_counter}/{args.patience}")
            ### TRAIN
            train_auc, train_acc, loss_avg = train(train_loader, model, optimizer, args, None)
            
            ### VALID
            valid_auc, valid_acc, preds, targets = validate(valid_loader, model, args)
            
            # wandb.log({"lr": optimizer.param_groups[0]['lr'], "train_loss": loss_avg, "train_auc": train_auc, "train_acc":train_acc,
            #             "valid_auc":valid_auc, "valid_acc":valid_acc})
            
            ### TODO: model save or early stopping
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_model = copy.deepcopy(model)
                early_stopping_counter = 0
            else :
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                    break
            # scheduler
            if args.scheduler == 'plateau':
                scheduler.step(best_auc)
            else:
                scheduler.step()

        return best_model

    def evaluate(self, args, model, valid_data):
        """훈련된 모델과 validation 데이터셋을 제공하면 predict 반환"""
        pin_memory = False

        valset = DKTDataset(valid_data, args, False)
        valid_loader = torch.utils.data.DataLoader(valset, shuffle=False,
                                                batch_size=args.batch_size,
                                                pin_memory=pin_memory,
                                                collate_fn=collate)

        auc, acc, preds, _ = validate(valid_loader, model, args)

        return preds

    def test(self, args, model, test_data):
        return self.evaluate(args, model, test_data)

    def get_target(self, datas):
        targets = []
        for data in datas:
            targets.append(data[-1][-1])

        return np.array(targets)
    
class PseudoLabel:
    def __init__(self, trainer, args):
        self.trainer = trainer
        self.args = args
        
        self.origin_model = None 
        
        self.model_path = os.path.join(args.model_dir, args.model_name)
        if os.path.exists(self.model_path):
            self.load_model()
        
        # 결과 저장용
        self.models =[]
        self.valid_aucs =[]
        self.valid_accs =[]

    def load_model(self):
        
        model_path = os.path.join(self.args.model_dir, self.args.model_name)
        print("Loading Model from:", model_path)
        load_state = torch.load(model_path)
        model = get_model(self.args)

        # 1. load model state
        model.load_state_dict(load_state['state_dict'], strict=True)
        
        print("Loading Model from:", model_path, "...Finished.")
        self.orgin_model = model

    def visualize(self):
        aucs = self.valid_aucs
        accs = self.valid_accs

        N = len(aucs)
        auc_min = min(aucs)
        auc_max = max(aucs)
        acc_min = min(accs)
        acc_max = max(accs)

        experiment = ['base'] + [f'pseudo {i + 1}' for i in range(N - 1)]
        df = pd.DataFrame({'experiment': experiment, 'auc': aucs, 'acc': accs})

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(5 + N, 7))

        ax1.set_title('AUC of Pseudo Label Training Process', fontsize=16)

        # Time
        plt.bar(df['experiment'],
                df['auc'],
                color='red',
                width=-0.3, align='edge',
                label='AUC')
        plt.ylabel('AUC (Area Under the ROC Curve)')
        ax1.set_ylim(auc_min - 0.002, auc_max + 0.002)
        ax1.axhline(y=aucs[0], color='r', linewidth=1)
        ax1.legend(loc=2)

        # AUC
        ax2 = ax1.twinx()
        plt.bar(df['experiment'],
                df['acc'],
                color='blue',
                width=0.3, align='edge',
                label='ACC')
        plt.ylabel('ACC (Accuracy)')

        ax2.grid(False)
        ax2.set_ylim(acc_min - 0.002, acc_max + 0.002)
        ax2.axhline(y=accs[0], color='b', linewidth=1)
        ax2.legend(loc=1)

        plt.show()

    def train(self, args, train_data, valid_data):
        model = self.trainer.train(args, train_data, valid_data)

        # model 저장
        self.models.append(model)
        
        return model

    def validate(self, args, model, valid_data):
        valid_target = self.trainer.get_target(valid_data)
        valid_predict = self.trainer.evaluate(args, model, valid_data)

        # Metric
        valid_auc, valid_acc = get_metric(valid_target, valid_predict)

        # auc / acc 저장
        self.valid_aucs.append(valid_auc)
        self.valid_accs.append(valid_acc)

        print(f'Valid AUC : {valid_auc} Valid ACC : {valid_acc}')

    def test(self, args, model, test_data):
        test_predict = self.trainer.test(args, model, test_data)
        pseudo_labels = np.where(test_predict >= 0.5, 1, 0)
        
        with open(args.write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(args.write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(test_predict):
                w.write('{},{}\n'.format(id,p))
                
        return pseudo_labels

    def update_train_data(self, pseudo_labels, train_data, test_data):
        # pseudo 라벨이 담길 test 데이터 복사본
        pseudo_test_data = copy.deepcopy(test_data)

        # pseudo label 테스트 데이터 update
        for test_data, pseudo_label in zip(pseudo_test_data, pseudo_labels):
            test_data[-1][-1] = pseudo_label
        
        # train data 업데이트
        pseudo_train_data = np.concatenate((train_data, pseudo_test_data))

        return pseudo_train_data
    
    def run(self, N, args, train_data, valid_data, test_data):
        """
        N은 두번째 과정을 몇번 반복할지 나타낸다.
        즉, pseudo label를 이용한 training 횟수를 가리킨다.
        """
        if N < 1:
            raise ValueError(f"N must be bigger than 1, currently {N}")
        
        # pseudo label training을 위한 준비 단계
        print("Preparing for pseudo label process")
        if self.origin_model :
            model = self.model
            self.models.append(model)
        else :
            model = self.train(args, train_data, valid_data)
        self.validate(args, model, valid_data)
        
        args.write_path = f'/opt/ml/pseudo/output_0.csv'
        
        pseudo_labels = self.test(args, model, test_data)
        pseudo_train_data = self.update_train_data(pseudo_labels, train_data, test_data)

        # pseudo label training 원하는 횟수만큼 반복
        for i in range(N):
            print(f'Pseudo Label Training Process {i + 1}')
            # seed
            seed_everything(args.seed)
            args.write_path = f'/opt/ml/pseudo/output_{i}.csv'
            model = self.train(args, pseudo_train_data, valid_data)
            self.validate(args, model, valid_data)
            pseudo_labels = self.test(args, model, test_data)
            pseudo_train_data = self.update_train_data(pseudo_labels, train_data, test_data)
            

        # 결과 시각화
        self.visualize()
    
def main(args):
    # wandb.login()
    
    setSeeds(args.seed)
    
    args = add_features(args)
    
    # args.cate_col_e = ["grade","KnowledgeTag","assessmentItemID","testId"]
    # args.cate_col_d = []
    # args.cont_col_e = ["ass_elp","ass_elp_o","ass_elp_x","prb_elp","prb_elp_o","prb_elp_x","test_mean","ass_mean","test_mean","prb_mean"]
    # args.cont_col_d = ["elapsed"]
    # args.n_cate_e = len(args.cate_col_e)
    # args.n_cate_d = len(args.cate_col_d)
    # args.n_cont_e = len(args.cont_col_e)
    # args.n_cont_d = len(args.cont_col_d)
    args.model_name = "pseudo.pt"
    wandb.config.update(args)
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    preprocess.load_test_data(args.test_file_name, is_train_ = True)
    
    data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data)
    test_data = preprocess.get_test_data()
    
    if args.sep_grade == True :
        ddf = test_data[test_data['answerCode']==-1]
        test2train_data = test_data[~test_data.set_index(['userID','grade']).index.isin(ddf.set_index(['userID','grade']).index)]
        train_data = pd.concat([train_data,test2train_data])
        
        test_data = test_data[test_data.set_index(['userID','grade']).index.isin(ddf.set_index(['userID','grade']).index)]

    
    # train_data = pd.concat([train_data[:10000],train_data[-10000:]])
    # valid_data = valid_data[:10000]
    # test_data = test_data[:10000]
    
    train_data = post_process(train_data, args)
    valid_data = post_process(valid_data, args)
    test_data = post_process(test_data, args)
    
    trainer = Trainer()
    pseudo = PseudoLabel(trainer, args)


    N = 5
    pseudo.run(N, args, train_data, valid_data, test_data)
    origin = "pseudo"
    
    args.model_dir = "/opt/ml/pseudo/model/"
    for i, model in enumerate(pseudo.models):
        model_to_save = model.module if hasattr(model, 'module') else model
        args.model_name = f"{origin}_{i}.pt"
        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model_to_save.state_dict(),
            },
            args.model_dir, args.model_name,
        )
    with open(os.path.join(args.model_dir,'arg.pkl'),'wb') as file:
        pickle.dump(args, file)
        
if __name__ == '__main__':
    wandb.login()
    wandb.init(entity = 'dkdkt', project='Pseudo')
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
