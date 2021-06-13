import re

# args.Tfixup 설정해줘야함
class TfixupSaint(nn.Module):
    def __init__(self, args):
        super(TfixupSaint, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.dropout = self.args.drop_out
        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_grade = nn.Embedding(self.args.n_grade + 1, self.hidden_dim // 3)

        self.enc_comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)
        # Decoder embed
        self.cate_proj = nn.Linear((self.hidden_dim // 3) * 1, self.hidden_dim)
        self.cont_bn = nn.BatchNorm1d(args.n_cont)
        self.cont_proj = nn.Linear(args.n_cont, self.hidden_dim)
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
        if self.args.Tfixup:

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
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param
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
                temp_state_dict[name] = (9 * (self.args.n_layers) ** (-1 / 4)) * param
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
        batch_size = input["interaction"].size(0)
        seq_len = input["interaction"].size(1)
        not_cont = [
            "interaction",
            "testId",
            "assessmentItemID",
            "KnowledgeTag",
            "grade",
            "answerCode",
            # "problem_interaction",
            # "tag_interaction",
            "mask",
            "gather_index",
        ]

        # 신나는 embedding
        embed_test = self.embedding_test(input["testId"])
        embed_question = self.embedding_question(input["assessmentItemID"])
        embed_tag = self.embedding_tag(input["KnowledgeTag"])
        embed_grade = self.embedding_grade(input["grade"])

        embed_enc = torch.cat(
            [embed_test, embed_question, embed_tag, embed_grade],
            2,
        )
        embed_enc = self.enc_comb_proj(embed_enc)
        # DECODER
        # Categorical
        embed_interaction = self.embedding_interaction(input["interaction"])
        cate_embed = torch.cat([embed_interaction], 2)
        cate_embed = self.cate_proj(cate_embed)
        # Continuous
        cont_emb = torch.cat(
            [v.unsqueeze(-1) for k, v in input.items() if k not in not_cont],
            dim=-1,
        )
        cont_emb = self.cont_bn(cont_emb.view(-1, cont_emb.size(-1))).view(
            batch_size, -1, cont_emb.size(-1)
        )
        cont_emb = self.cont_proj(cont_emb)

        embed_dec = torch.cat([cate_embed, cont_emb], 2)
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
