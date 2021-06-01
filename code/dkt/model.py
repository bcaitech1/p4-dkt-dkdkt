from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        indvi_dim = len(self.args.cate_cols)+len(self.args.non_cate_cols)
        if not indvi_dim:
            raise RuntimeError("no feature found.")
        if self.args.cate_cols:
            self.embedding_interaction = nn.Embedding(
                3, self.hidden_dim//indvi_dim)
            self.embedding_category = {}
            for col_name, col_len in self.args.cate_cols.items():
                self.embedding_category[col_name] = nn.Embedding(
                    col_len + 1, self.hidden_dim//indvi_dim)
                setattr(self, f'emb_{col_name}',
                        self.embedding_category[col_name])

            # embedding combination projection
            self.comb_proj = nn.Sequential(
                nn.Linear((self.hidden_dim//indvi_dim)*(len(self.args.cate_cols)+1),
                          (self.hidden_dim*len(self.args.cate_cols))//indvi_dim),
                nn.LayerNorm(
                    (self.hidden_dim*len(self.args.cate_cols))//indvi_dim)
            )
        if self.args.non_cate_cols:
            self.linear_non_cate = nn.Sequential(
                nn.Linear(len(self.args.non_cate_cols),
                          (self.hidden_dim*len(self.args.non_cate_cols))//indvi_dim),
                nn.LayerNorm(
                    (self.hidden_dim*len(self.args.non_cate_cols))//indvi_dim)
            )

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

        batch_size = input['interaction'].size(0)
        # Embedding

        embed_interaction = self.embedding_interaction(input['interaction'])
        feature_linear = []

        if self.args.cate_cols:
            cate_embed_list = []
            for k in self.embedding_category.keys():
                cate_embed_list.append(self.embedding_category[k](input[k]))

            embed_list = [embed_interaction] + cate_embed_list
            cate_embed = torch.cat(embed_list, 2)

            feature_linear.append(self.comb_proj(cate_embed))

        if self.args.non_cate_cols:
            cont_col_list = []
            for cont_col in self.args.non_cate_cols:
                cont_col_list.append(torch.unsqueeze(input[cont_col], 2))
            cont_all = torch.cat(cont_col_list, 2)

            feature_linear.append(self.linear_non_cate(cont_all))

        X = torch.cat(feature_linear, 2)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds
