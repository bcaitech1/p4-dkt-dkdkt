import os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pickle
import wandb
import re
try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
import matplotlib.pyplot as plt
from dkt.utils import duplicate_name_changer


class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        cate_col_num = len(self.args.cate_cols.keys())
        cont_col_num = len(self.args.cont_cols)
        divider = (bool(cate_col_num) + bool(cont_col_num))
        if divider == 0:
            raise RuntimeError("no feature found.")
        self.embedding_interaction = nn.Embedding(
            3, self.hidden_dim//3)
        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        if self.args.cate_cols:
            self.embedding_category = {}
            for col_name, col_len in self.args.cate_cols.items():
                self.embedding_category[col_name] = nn.Embedding(
                    col_len + 1, self.hidden_dim//3)
                setattr(self, f'emb_{col_name}',
                        self.embedding_category[col_name])

        # embedding combination projection
        self.cate_proj = nn.Sequential(
            nn.Linear((self.hidden_dim//3)*(cate_col_num+1),
                      self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        if self.args.cont_cols:
            self.bn_cont = nn.BatchNorm1d(cont_col_num)
            self.embedding_cont = nn.Sequential(
                nn.Linear(cont_col_num,
                          self.hidden_dim),
                nn.LayerNorm(
                    self.hidden_dim
                )
            )
        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim *
                      divider, self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim))

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input: Dict):

        batch_size = input['oth']['interaction'].size(0)
        # Embedding

        embed_interaction = self.embedding_interaction(
            input['oth']['interaction'])

        feature_linear = []

        cate_embed_list = []
        if self.args.cate_cols:
            for k in self.embedding_category.keys():
                cate_embed_list.append(
                    self.embedding_category[k](input['cate'][k]))

        embed_list = [embed_interaction] + cate_embed_list
        cate_embed = torch.cat(embed_list, 2)

        feature_linear.append(self.cate_proj(cate_embed))

        if self.args.cont_cols:
            cont_col_list = []
            for cont_col in self.args.cont_cols:
                cont_col_list.append(input['cont'][cont_col].unsqueeze(2))
            cont_all = torch.cat(cont_col_list, 2)
            cont = self.bn_cont(
                cont_all.view(-1, cont_all.size(-1))).view(batch_size, -1, cont_all.size(-1))

            feature_linear.append(self.embedding_cont(cont))

        X = self.comb_proj(torch.cat(feature_linear, 2))

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """

    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim

        cate_col_num = len(self.args.cate_cols.keys())
        cont_col_num = len(self.args.cont_cols)
        divider = (bool(cate_col_num) + bool(cont_col_num))
        if divider == 0:
            raise RuntimeError("no feature found.")
        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(
            3, self.hidden_dim//3)

        if self.args.cate_cols:
            self.embedding_category = {}
            for col_name, col_len in self.args.cate_cols.items():
                self.embedding_category[col_name] = nn.Embedding(
                    col_len + 1, self.hidden_dim//3)
                setattr(self, f'emb_{col_name}',
                        self.embedding_category[col_name])

        # embedding combination projection
        self.cate_proj = nn.Sequential(
            nn.Linear((self.hidden_dim//3)*(cate_col_num+1),
                      self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        if self.args.cont_cols:
            self.bn_cont = nn.BatchNorm1d(cont_col_num)
            self.embedding_cont = nn.Sequential(
                nn.Linear(cont_col_num, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim))

        # embedding combination projection
        self.comb_proj = nn.Sequential(
            nn.Dropout(self.args.drop_out),
            nn.Linear(self.hidden_dim*divider, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim))

        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)

        # Encoder
        self.query = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )
        self.key = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )
        self.value = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.args.n_heads
        )
        self.mask = None  # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.args.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.args.n_layers, batch_size, self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.args.n_layers, batch_size, self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        # Categorical Variable Embedding

        batch_size = input['oth']['interaction'].size(0)
        # Embedding

        embed_interaction = self.embedding_interaction(
            input['oth']['interaction'])

        feature_linear = []

        cate_embed_list = []
        if self.args.cate_cols:
            for k in self.embedding_category.keys():
                cate_embed_list.append(
                    self.embedding_category[k](input['cate'][k]))

        embed_list = [embed_interaction] + cate_embed_list
        cate_embed = torch.cat(embed_list, 2)

        feature_linear.append(self.cate_proj(cate_embed))

        # continuous variable embedding
        # batch normalization
        if self.args.cont_cols:
            cont_col_list = []
            for cont_col in self.args.cont_cols:
                cont_col_list.append(input['cont'][cont_col].unsqueeze(2))
            cont_all = torch.cat(cont_col_list, 2)
            cont = self.bn_cont(
                cont_all.view(-1, cont_all.size(-1))).view(batch_size, -1, cont_all.size(-1))
            embed_cont = self.embedding_cont(cont)

            feature_linear.append(embed_cont)

        # Running LSTM
        embed = self.comb_proj(torch.cat(feature_linear, 2))

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        # attention
        # last query only
        out, _ = self.attn(q, k, v)

        # residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        # feed forward network
        out = self.ffn(out)

        # residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):

    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        cate_col_num = len(self.args.cate_cols.keys())
        cont_col_num = len(self.args.cont_cols)
        divider = (bool(cate_col_num) + bool(cont_col_num))
        if divider == 0:
            raise RuntimeError("no feature found.")
        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(
            3, self.hidden_dim//3)

        if self.args.cate_cols:
            self.embedding_category = {}
            for col_name, col_len in self.args.cate_cols.items():
                self.embedding_category[col_name] = nn.Embedding(
                    col_len + 1, self.hidden_dim//3)
                setattr(self, f'emb_{col_name}',
                        self.embedding_category[col_name])

        # embedding combination projection
        self.cate_proj = nn.Sequential(
            nn.Linear((self.hidden_dim//3)*(cate_col_num+1),
                      self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        if self.args.cont_cols:
            self.bn_cont = nn.BatchNorm1d(cont_col_num)
            self.embedding_cont = nn.Sequential(
                nn.Linear(cont_col_num, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim))

        # embedding combination projection
        self.comb_proj = nn.Sequential(
            nn.Dropout(self.args.drop_out),
            nn.Linear(self.hidden_dim*divider, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim))

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        # Categorical Variable Embedding
        batch_size = input['oth']['interaction'].size(0)
        # Embedding

        embed_interaction = self.embedding_interaction(
            input['oth']['interaction'])

        feature_linear = []

        cate_embed_list = []
        if self.args.cate_cols:
            for k in self.embedding_category.keys():
                cate_embed_list.append(
                    self.embedding_category[k](input['cate'][k]))

        embed_list = [embed_interaction] + cate_embed_list
        cate_embed = torch.cat(embed_list, 2)

        feature_linear.append(self.cate_proj(cate_embed))

        # continuous variable embedding
        # batch normalization
        if self.args.cont_cols:
            cont_col_list = []
            for cont_col in self.args.cont_cols:
                cont_col_list.append(input['cont'][cont_col].unsqueeze(2))
            cont_all = torch.cat(cont_col_list, 2)
            cont = self.bn_cont(
                cont_all.view(-1, cont_all.size(-1))).view(batch_size, -1, cont_all.size(-1))
            embed_cont = self.embedding_cont(cont)

            feature_linear.append(embed_cont)

        # Running LSTM
        embed = self.comb_proj(torch.cat(feature_linear, 2))
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(embed, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = input['oth']["mask"].unsqueeze(
            1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(
            out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        # if self.args.loss == 'arcface':
        #     sequence_output = sequence_output[:, -1,
        #                                       :].contiguous().view(batch_size, -1)
        #     return sequence_output

        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class LGBM:
    def __init__(self, args):
        self.args = args

    def train(self, train, valid, test, args):
        result = {"epoch": 0, "train_loss": 0, "train_auc": 0, "train_acc": 0,
                  "valid_auc": 0.7, "valid_acc": 0.7}
        wandb.log(result)
        # X, y 값 분리
        y_train = train['answerCode'].values.ravel()
        train = train.drop(['answerCode'], axis=1)

        y_valid = valid['answerCode'].values.ravel()
        valid = valid.drop(['answerCode'], axis=1)
        lgb_train = lgb.Dataset(train[train.columns], y_train)
        lgb_valid = lgb.Dataset(valid[valid.columns], y_valid)
        os.makedirs(os.path.join('./models/', args.model_alias), exist_ok=True)
        folder= os.path.join('./models/', args.model_alias)
        output = os.path.join(folder, "model.txt")
        
        args.seed = 74
        args.extra_trees = False
        args.lr = 0.041015616417097805
        args.xgb_dart = True
        args.num_leaves = 62
        args.drop_out = 0.05177613641421279
        param =  {
                'tree_learner': args.tl,
                'seed': args.seed,
                'drop_seed': args.seed,
                'objective': 'binary',
                'metric': 'auc',
                'boosting': args.boosting,
                'num_threads': args.num_workers,
                'extra_trees': args.extra_trees,
                'drop_rate': args.drop_out,
                'xgboost_dart_mode': args.xgb_dart,
                'output_model': output,
                'learning_rate': args.lr,
                'device_type': "cpu",
                # 'max_bin': args.max_bin,
                'num_leaves': args.num_leaves
            }
        print(param)
        model = lgb.train(
            {
                'tree_learner': args.tl,
                'seed': args.seed,
                'drop_seed': args.seed,
                'objective': 'binary',
                'metric': 'auc',
                'boosting': args.boosting,
                'num_threads': args.num_workers,
                # 'extra_trees': args.extra_trees,
                'drop_rate': args.drop_out,
                'xgboost_dart_mode': args.xgb_dart,
                'output_model': output,
                'learning_rate': args.lr,
                'device_type': "cpu",
                # 'max_bin': args.max_bin,
                'num_leaves': args.num_leaves
            },
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            verbose_eval=10,
            num_boost_round=20,
            early_stopping_rounds=200,
        )
        model.save_model(output)
        preds = model.predict(valid[valid.columns])
        acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_valid, preds)
        print(f'VALID AUC : {auc} ACC : {acc}\n')
        result = {"epoch": self.args.n_epochs-1, "train_loss": 0, "train_auc": 0, "train_acc": 0,
                  "valid_auc": auc, "valid_acc": acc}
        wandb.log(result)

        # ax = lgb.plot_importance(model)
        # fig = ax.figure
        # fig.set_size_inches(30, 40)
        # output_name = duplicate_name_changer(
        #     f'./output/', f'{self.args.model}{self.args.save_suffix}')
        # os.makedirs(os.path.join('./output/', output_name), exist_ok=True)
        # plt.savefig(os.path.join(os.path.join(
        #     './output/', output_name), 'impor.png'))
        # LEAVE LAST INTERACTION ONLY
        # test = test[test['userID'] != test['userID'].shift(-1)]
        # # DROP ANSWERCODE
        # test = test.drop(['answerCode'], axis=1)

        total_preds = model.predict(test[test.columns])
        # SAVE OUTPUT
        output_name = duplicate_name_changer(
        self.args.output_dir, f"final.csv")
        write_path = os.path.join(os.path.join('./output/', args.model_alias), output_name)
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open('./final.csv', 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write('{},{}\n'.format(id,p))


class TfixupSaint(nn.Module):
    def __init__(self, args):
        super(TfixupSaint, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.dropout = self.args.drop_out

        cate_col_num = len(self.args.cate_cols.keys())
        cont_col_num = len(self.args.cont_cols)
        divider = (bool(cate_col_num) + bool(cont_col_num))
        if divider == 0:
            raise RuntimeError("no feature found.")

        # encoder
        if self.args.cate_cols:
            self.embedding_category = {}
            for col_name, col_len in self.args.cate_cols.items():
                self.embedding_category[col_name] = nn.Embedding(
                    col_len + 1, self.hidden_dim//3)
                setattr(self, f'embedding_{col_name}',
                        self.embedding_category[col_name])
        self.enc_comb_proj = nn.Linear(
            (self.hidden_dim // 3) * cate_col_num, self.hidden_dim)

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        if "tfixup" in self.args.model.lower():
            self.cate_proj = nn.Linear((self.hidden_dim//3)*1,
                                       self.hidden_dim)
        else:
            self.cate_proj = nn.Sequential(
                nn.Linear((self.hidden_dim//3)*1,
                          self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )

        # Decoder embed

        if self.args.cont_cols:
            self.cont_bn = nn.BatchNorm1d(cont_col_num)
            if "tfixup" in self.args.model.lower():
                self.cont_proj = nn.Linear(cont_col_num, self.hidden_dim)
            else:
                self.cont_proj = nn.Sequential(
                    nn.Linear(cont_col_num, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim)
                )

        self.comb_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            self.hidden_dim, self.dropout, self.args.max_seq_len
        )
        self.pos_decoder = PositionalEncoding(
            self.hidden_dim, self.dropout, self.args.max_seq_len
        )
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers,
            num_decoder_layers=self.args.n_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation="relu",
        )

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

        # T-Fixup
        if "tfixup" in self.args.model.lower():
            print("#######tfixup start!######")
            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixupbb Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0

        for name, param in self.named_parameters():
            if re.match(r"^embedding*", name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r".*ln.*|.*bn.*", name):
                continue
            elif re.match(r".*norm.*", name):
                continue
            elif re.match(r".*weight*", name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)

    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():

            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            # print(name)

            if re.match(r"^embedding*", name):
                temp_state_dict[name] = (
                    9 * self.args.n_layers) ** (-1 / 4) * param
            elif re.match(r".*norm.*", name):
                continue
            elif re.match(r".*ln.*|.*bn.*", name):
                continue
            elif re.match(r"encoder.*linear.*weight|encoder.*attn.*out.*weight", name):
                temp_state_dict[name] = (
                    0.67 * (self.args.n_layers) ** (-1 / 4)
                ) * param
            elif re.match(r"encoder.*attn.*in.*weight", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (
                    param * (2 ** 0.5)
                )
            elif re.match(r"decoder.*linear.*weight|decoder.*attn.*out.*weight", name):
                temp_state_dict[name] = (
                    9 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r"decoder.*attn.*in.*weight", name):
                temp_state_dict[name] = (9 * (self.args.n_layers) ** (-1 / 4)) * (
                    param * (2 ** 0.5)
                )

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, input):
        batch_size = input['oth']["interaction"].size(0)
        seq_len = input['oth']["interaction"].size(1)
        embed_interaction = self.embedding_interaction(
            input['oth']["interaction"])
        feature_linear = []

        cate_embed_list = []
        if self.args.cate_cols:
            for k in self.embedding_category.keys():
                cate_embed_list.append(
                    self.embedding_category[k](input['cate'][k]))

        embed_enc = torch.cat(cate_embed_list, 2)

        embed_enc = self.enc_comb_proj(embed_enc)

        # DECODER
        # Categorical
        cate_embed = torch.cat([embed_interaction], 2)
        feature_linear.append(self.cate_proj(cate_embed))
        # Continuous
        if self.args.cont_cols:
            cont_col_list = []
            for cont_col in self.args.cont_cols:
                cont_col_list.append(input['cont'][cont_col].unsqueeze(-1))
            cont_all = torch.cat(cont_col_list, -1)
            cont = self.cont_bn(
                cont_all.view(-1, cont_all.size(-1))).view(batch_size, -1, cont_all.size(-1))
            embed_cont = self.cont_proj(cont)

            feature_linear.append(embed_cont)

        embed_dec = torch.cat(feature_linear, 2)
        embed_dec = self.comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)

        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)

        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)

        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)

        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)

        out = self.transformer(
            embed_enc,
            embed_dec,
            src_mask=self.enc_mask,
            tgt_mask=self.dec_mask,
            memory_mask=self.enc_dec_mask,
        )

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        out = out.view(batch_size, -1)

        preds = self.activation(out)

        return preds


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)
