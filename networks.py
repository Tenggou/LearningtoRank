import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


def mask_(seq, seq_lens):
    mask = torch.zeros_like(seq)
    for i, l in enumerate(seq_lens):
        mask[i, :l].fill_(1)
    return mask


def mask_fill(seq, mask, fill_value):
    return seq.masked_fill(mask == 0, fill_value)


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM
    """

    def __init__(self, emb_dim=300, hidden_dim=300, dropout=0.0, device='cpu', emb_values=None):
        super(BiLSTM, self).__init__()
        self.device = device

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(emb_values, dtype=torch.float))
        self.embedding.weight.requires_grad = True

        self.rnn = nn.LSTM(input_size=self.emb_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=1,
                           bidirectional=True,  #
                           batch_first=True)  #
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # self.relu = nn.ReLU()

        self.init_h = nn.Parameter(torch.Tensor(2, self.hidden_dim))  # 2 为rnn层数*双向
        self.init_c = nn.Parameter(torch.Tensor(2, self.hidden_dim))
        INI = 1e-2
        torch.nn.init.uniform_(self.init_h, -INI, INI)  # 均匀分布
        torch.nn.init.uniform_(self.init_c, -INI, INI)

    def forward(self, x):
        """
        :param len: tensor, the lengths of x
        :param x: input  : batch sequences
        :return:
        """
        h = (self.init_h.unsqueeze(1).expand(2, len(x), self.hidden_dim).contiguous(),
             self.init_c.unsqueeze(1).expand(2, len(x), self.hidden_dim).contiguous())
        # h = (torch.ones(2, len(x), self.hidden_dim).to(device=self.device),
        #      torch.ones(2, len(x), self.hidden_dim).to(device=self.device))

        # 2019.9.9 how to use pack_padded_sequence
        lengths = torch.tensor([len(node) for node in x], dtype=torch.long).to(device=self.device)  # 未排序的序列长度
        x = [
            np.pad(x_node, (0, int(torch.max(lengths).item() - len(x_node))), mode='constant').tolist() for x_node in x
        ]
        x = torch.tensor(x, dtype=torch.long).to(device=self.device)
        x_embed = self.embedding(x)
        x_embed = self.dropout_layer(x_embed)

        lengths_sort, index_sort = torch.sort(lengths, descending=True)  # 排序后的结果，让原arr排序的index序列
        _, index_unsort = torch.sort(index_sort)  # 变回原arr的index序列
        x_embed = x_embed.index_select(0, index_sort)
        h = (h[0].index_select(1, index_sort), h[1].index_select(1, index_sort))

        x_embed = torch.nn.utils.rnn.pack_padded_sequence(x_embed, lengths_sort, batch_first=True)
        out, h_out = self.rnn(x_embed, h)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        unsort_out = out.index_select(0, index_unsort)
        masked_out = mask_fill(unsort_out, mask_(unsort_out, lengths), fill_value=-1e18)

        h_out = h_out[0].index_select(1, index_unsort).transpose(1, 0).contiguous().view(-1, self.hidden_dim * 2)

        # bilstm dense
        # return self.relu(self.linear(self.dropout_layer(
        #     F.adaptive_max_pool2d(unsort_out, (1, self.hidden_dim * 2)).view(-1, self.hidden_dim * 2)))), h_out

        # bilstm
        return F.adaptive_max_pool2d(masked_out, (1, self.hidden_dim * 2)).view(-1, self.hidden_dim * 2), h_out, \
               unsort_out, lengths


class DAM(nn.Module):
    """
    Decomposable Attention Model
    """
    def __init__(self, emb_dim=300, hidden_dim=300, dropout=0.0, device='cpu', emb_values=None):
        super(DAM, self).__init__()
        self.device = device

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.encoder = BiLSTM(emb_dim=self.emb_dim, hidden_dim=self.hidden_dim, dropout=self.dropout,
                              device=self.device, emb_values=emb_values).to(device=self.device)

        self.F = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False),
            nn.ReLU()
        )
        self.G = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2, bias=False)
        self.H = nn.Linear(self.hidden_dim * 4, 1, bias=False)

    def forward(self, q, p):
        """
        取bilstm的输出作为dam的输入
        :param q:
        :param p:
        :return:
        """
        _, _, a, a_lens = self.encoder(q)
        _, _, b, b_lens = self.encoder(p)

        mask_a = mask_(a, a_lens)
        mask_b = mask_(b, b_lens)
        # masked_a = mask_fill(a, mask_a, fill_value=0)
        # masked_b = mask_fill(b, mask_b, fill_value=0)
        # F函数
        att_a = self.F(a.view(-1, self.hidden_dim * 2)).view(a.shape[0], a.shape[1], self.hidden_dim * 2)
        att_b = self.F(b.view(-1, self.hidden_dim * 2)).view(b.shape[0], b.shape[1], self.hidden_dim * 2)

        # e[i][j]
        e = torch.bmm(att_a, att_b.transpose(2, 1))
        mask_a_b = torch.bmm(mask_a, mask_b.transpose(2, 1))
        masked_e = mask_fill(e, mask_a_b, fill_value=-1e18)  # exp(-1e18) 约等于0

        alpha = torch.bmm(F.softmax(masked_e, dim=-2).transpose(2, 1), a)  # b模仿a  b_len,self.hidden_dim*2
        beta = torch.bmm(F.softmax(masked_e, dim=-1), b)  # a模仿b   a_len, self.hidden_dim*2
        # 这里要mask_fill，和a, b一致
        alpha = mask_fill(alpha, mask_(alpha, b_lens), fill_value=0)
        beta = mask_fill(beta, mask_(beta, a_lens), fill_value=0)
        # sum会减少一维
        v1 = torch.sum(self.G(torch.cat((a, beta), dim=-1).view(-1, self.hidden_dim * 4)).view(a.shape[0], a.shape[1],
                                                                                               self.hidden_dim * 2),
                       dim=1)
        v2 = torch.sum(self.G(torch.cat((b, alpha), dim=-1).view(-1, self.hidden_dim * 4)).view(b.shape[0], b.shape[1],
                                                                                                self.hidden_dim * 2),
                       dim=1)
        return self.H(torch.cat((v1, v2), dim=-1)).view(-1)


class SMM_q(nn.Module):
    """
    Slot Matching Model, to encoder question
    """

    def __init__(self, emb_dim=300, hidden_dim=300, dropout=0.0, device='cpu', emb_values=None):
        super(SMM_q, self).__init__()
        self.device = device

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.lstm = BiLSTM(emb_dim=self.emb_dim, hidden_dim=self.hidden_dim, dropout=self.dropout,
                           device=self.device, emb_values=emb_values).to(device=self.device)
        self.k = nn.Linear(self.hidden_dim * 2, 2, bias=False)
        # self.linear = nn.Linear(self.hidden_dim, self.hidden_dim * 2, bias=False)
        # self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, q):
        q_out, _, Q, Q_len = self.lstm(q)
        masked_Q = mask_fill(Q, mask_(Q, Q_len), fill_value=0)

        alpha = F.softmax(self.k(masked_Q.view(-1, self.hidden_dim * 2)).view(masked_Q.shape[0], masked_Q.shape[1], 2),
                          dim=1)  # dim=1 意味seq每一个词的attention（重要程度） self-attention???
        # dim=-1 意味seq对hop的

        # q_emb = self.lstm.embedding(self.pad_seq(q))
        # q_emb = q_emb.repeat(1, 1, 2)  # 如果用linear
        # q_emb = self.linear(q_emb.view(-1, self.hidden_dim)).view(q_emb.shape[0], q_emb.shape[1], self.hidden_dim * 2)
        Q_ = torch.bmm(alpha.transpose(2, 1), masked_Q)  # + q_emb)  # batch size, 2, hidden_dim * 2

        # q_out = torch.cat((Q_[:, 0, :], Q_[:, 1, :]), dim=-1)
        q_out = F.adaptive_max_pool2d(Q_, (1, self.hidden_dim * 2)).view(-1, self.hidden_dim * 2)
        return q_out

    def pad_seq(self, x):
        lengths = torch.tensor([len(node) for node in x], dtype=torch.long).to(device=self.device)  # 未排序的序列长度
        x = [
            np.pad(x_node, (0, int(torch.max(lengths).item() - len(x_node))), mode='constant').tolist() for x_node in x
        ]
        x = torch.tensor(x, dtype=torch.long).to(device=self.device)
        return x


class SMM_p(nn.Module):
    """
    Slot Matching Model, to encoder question
    """

    def __init__(self, emb_dim=300, hidden_dim=300, dropout=0.0, device='cpu', emb_values=None):
        super(SMM_p, self).__init__()
        self.device = device

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.lstm = BiLSTM(emb_dim=self.emb_dim, hidden_dim=self.hidden_dim, dropout=self.dropout,
                           device=self.device, emb_values=emb_values).to(device=self.device)
        # self.linear = nn.Linear(self.hidden_dim, self.hidden_dim * 2, bias=False)
        # self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, c1, c2):

        c1_enc, _, _, _ = self.lstm(c1)  # batch size, 2, hidden_dim * 2
        # c1_emb = self.lstm.embedding(self.pad_seq(c1)[0])
        # c1_emb = c1_emb.repeat(1, 1, 2)  # linear
        # c1_emb = self.linear(c1_emb.view(-1, self.hidden_dim)).view(c1_emb.shape[0], c1_emb.shape[1],
        #                                                             self.hidden_dim * 2)
        c1 = c1_enc  # + torch.mean(c1_emb, dim=1)

        c2_lens = torch.tensor([len(node) for node in c2], dtype=torch.long).to(device=self.device)  # 未排序的序列长度
        sorted_c2_lens, sorted_index = torch.sort(c2_lens, descending=True)
        _, unsorted_index = torch.sort(sorted_index, descending=False)
        if torch.max(sorted_c2_lens).item() == 0:
            return c1
        if torch.min(sorted_c2_lens).item() == 0:
            c2 = [c2[i] for i in sorted_index]  # 按照长度排序
            right = sorted_c2_lens.tolist().index(0)
            c2_enc = self.lstm(c2[:right])[0]
            # c2 = torch.cat((c2_enc, torch.zeros(len(c2)-right, self.hidden_dim*2).to(self.device)), dim=0)  # sorted
            c2 = torch.cat((c2_enc, torch.Tensor([[-1e18]*self.hidden_dim*2]*(len(c2)-right)).to(self.device)), dim=0)  # sorted
            c2 = c2.index_select(0, unsorted_index)
            # print(sorted_c2_lens.index_select(0, unsorted_index)[0])
            # print(c2_lens[0])
        else:
            c2 = self.lstm(c2)[0]
        c_out = torch.cat((c1.unsqueeze(1), c2.unsqueeze(1)), dim=1)
        c_out = F.adaptive_max_pool2d(c_out, (1, self.hidden_dim * 2)).view(-1, self.hidden_dim * 2)
        # print(self.lstm.embedding(torch.LongTensor([0]).to(self.device))[:5])
        return c_out
