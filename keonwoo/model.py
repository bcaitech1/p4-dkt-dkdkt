import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
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


class Last_Query_Transformer(nn.Module):
    """
    Embedding --> MLH --> LSTM
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim)
        self.embedding_elapsed = nn.Embedding(self.args.n_elapsed + 1, self.hidden_dim)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim) * 5, self.hidden_dim)

        self.multi_en = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.n_heads, dropout=self.drop_out
        )  # multihead attention    ## todo add dropout, LayerNORM
        self.ffn_en = Feed_Forward_block(
            self.hidden_dim
        )  # feedforward block     ## todo dropout, LayerNorm
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
        )

        self.out = nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, input):

        test, question, tag, _, elapsed, mask, interaction, _ = input
        batch_size = interaction.size(0)

        embed_interaction = self.embedding_interaction(interaction)
        embed_interaction = nn.Dropout(self.drop_out)(embed_interaction)

        embed_test = self.embedding_test(test)
        embed_test = nn.Dropout(self.drop_out)(embed_test)

        embed_question = self.embedding_question(question)
        embed_question = nn.Dropout(self.drop_out)(embed_question)

        embed_tag = self.embedding_tag(tag)
        embed_tag = nn.Dropout(self.drop_out)(embed_tag)

        embed_elapsed = self.embedding_elapsed(elapsed)
        embed_elapsed = nn.Dropout(self.drop_out)(embed_elapsed)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_elapsed,
            ],
            2,
        )
        out = self.comb_proj(embed)

        out = out.permute(1, 0, 2)  # (n,b,d)  # print('pre multi', out.shape )

        # Multihead attention
        n, _, _ = out.shape
        out = self.layer_norm1(out)  # Layer norm
        skip_out = out

        out, attn_wt = self.multi_en(out[-1:, :, :], out, out)  # Q,K,V
        #                        #attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        # print('MLH out shape', out.shape)
        out = out + skip_out  # skip connection

        # LSTM
        out, _ = self.lstm(out)  # seq_len, batch, input_size
        out = out[-1:, :, :]

        # feed forward
        out = out.permute(1, 0, 2)  # (b,n,d)
        out = self.layer_norm2(out)  # Layer norm
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out  # skip connection

        out = self.out(out)
        out = self.activation(out)

        return out.squeeze(-1), 0
