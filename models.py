import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
import math
import random

torch.manual_seed(102)
torch.cuda.manual_seed_all(102)
np.random.seed(102)
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from transformer import *
from tqdm import tqdm
from components import *
from Informer.models.model import *


def attention_net(x, query, mask=None):
    d_k = query.size(-1)  # d_k为query的维度
    scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
    # 对最后一个维度 归一化得分
    alpha_n = F.softmax(scores, dim=-1)
    context = torch.matmul(alpha_n, x).sum(1)

    return context, alpha_n


class ATT(nn.Module):
    def __init__(self, units=64):
        super(ATT, self).__init__()
        self.norm = nn.BatchNorm1d(units, affine=False)
        self.Q = nn.Linear(units, units)
        self.K = nn.Linear(units, units)
        self.V = nn.Linear(units, units)
        self.fuse = nn.Linear(units * 2, units)
        self.units = units

        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.fuse.weight)

    def forward(self, query, keys, norm=False):
        p = []
        for k in range(keys.shape[1]):
            p.append(torch.matmul(self.Q(query).unsqueeze(1),
                                  self.K(keys[:, k, :]).unsqueeze(1).permute(0, 2, 1)) / math.sqrt(
                self.units))
        p = F.softmax(torch.cat(p, dim=-1), dim=-1)
        # print("attention score: ")
        # print(p)
        # a=query + torch.matmul(p, self.V(keys)).squeeze(1)
        if norm:
            return self.norm(query + torch.matmul(p, self.V(keys)).squeeze(1))
        else:
            return F.relu(self.fuse(torch.cat([torch.matmul(p, self.V(keys)).squeeze(1), query], dim=-1)))


class naive_kge_ts_model(nn.Module):
    def __init__(self, units=64, input_shape1=None, num_node=2, mode='double_lstm', device="cuda:0"):
        super(naive_kge_ts_model, self).__init__()
        self.input_shape1 = input_shape1
        self.num_node = num_node

        self.units = units

        self.mlp = nn.Sequential(
            # nn.Linear(units * (num_node + 1), units // 2),
            # nn.Dropout(p=0.1),
            # nn.ReLU(),
            nn.Linear(units * (num_node + 1), 1)
        )

        for m in self.mlp.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        self.lstm = nn.LSTM(input_size=input_shape1, hidden_size=units, num_layers=2, bidirectional=False,
                            batch_first=True, dropout=0.1)
        self.lstm2 = nn.ModuleList(
            nn.LSTM(input_size=input_shape1, hidden_size=units, num_layers=2, bidirectional=False,
                    batch_first=True, dropout=0.1) for i in range(num_node))
        # self.lstm2 = [e.to(device) for e in self.lstm2]

        self.criterion = nn.SmoothL1Loss()

    def forward(self, ts, gts):
        gts = gts.transpose(0, 1)

        _, (h_n, c_n) = self.lstm(ts)
        # print(h_n.shape)
        temp = []
        for i in range(gts.shape[0]):
            _, (h_n_g, c_n_g) = self.lstm2[i](gts[i, :, :, :])
            # print(h_n_g)
            temp.append(h_n_g.squeeze())

        # logits = self.mlp(h_n.squeeze())
        logits = self.mlp(torch.cat([h_n.squeeze()] + temp, dim=-1))
        # print(logits.shape)
        return logits  # F.sigmoid(logits)


class auto_fusion_kge_ts_model(nn.Module):
    def __init__(self, units=64, input_shape1=None, num_node=2, seq_len=11, type_encoder='gru', device="cuda:0"):
        super(auto_fusion_kge_ts_model, self).__init__()
        self.input_shape1 = input_shape1
        self.num_node = num_node

        self.units = units
        self.type_encoder = type_encoder

        self.fusion = nn.Linear(num_node + 1, 1)
        nn.init.xavier_uniform_(self.fusion.weight)

        self.mlp = nn.Sequential(
            nn.Linear(units, units // 2),
            nn.ReLU(),
            nn.Linear(units // 2, 1))

        for m in self.mlp.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        if type_encoder == 'lstm':
            self.encoder = nn.LSTM(input_size=input_shape1, hidden_size=units, num_layers=2, bidirectional=False,
                                   batch_first=True,
                                   dropout=0)
            self.encoder2 = nn.ModuleList(
                nn.LSTM(input_size=input_shape1, hidden_size=units, num_layers=2, bidirectional=False,
                        batch_first=True, dropout=0) for i in range(num_node))
        elif type_encoder == 'gru':
            self.encoder = nn.GRU(input_size=input_shape1, hidden_size=units, num_layers=2, bidirectional=False,
                                  batch_first=True,
                                  dropout=0)
            self.encoder2 = nn.ModuleList(
                nn.GRU(input_size=input_shape1, hidden_size=units, num_layers=2, bidirectional=False,
                       batch_first=True, dropout=0) for i in range(num_node))

        self.criterion = nn.SmoothL1Loss()

    def forward(self, ts, gts):
        gts = gts.transpose(0, 1)
        if self.type_encoder == 'lstm':
            _, (h_n, c_n) = self.encoder(ts)
            # print(h_n[-1, :, :].shape)

            temp = []
            for i in range(gts.shape[0]):
                _, (h_n_g, c_n_g) = self.encoder2[i](gts[i, :, :, :])
                temp.append(h_n_g[-1, :, :].unsqueeze(0))

            logits = self.fusion(torch.cat([h_n[-1, :, :].unsqueeze(0)] + temp, dim=0).permute(1, 2, 0))
        elif self.type_encoder == 'gru':
            _, h_n = self.encoder(ts)
            # print(h_n[-1, :, :].shape)

            temp = []
            for i in range(gts.shape[0]):
                _, h_n_g = self.encoder2[i](gts[i, :, :, :])
                temp.append(h_n_g[-1, :, :].unsqueeze(0))

            logits = self.fusion(torch.cat([h_n[-1, :, :].unsqueeze(0)] + temp, dim=0).permute(1, 2, 0))
        logits = self.mlp(logits.squeeze(-1))
        # print(logits.shape)
        return logits  # F.sigmoid(logits)


class attention_ts(nn.Module):
    def __init__(self, units=64, num_node=2, input_shape1=None, seq_len=11, device="cuda:0", type_encoder='lstm'):
        super(attention_ts, self).__init__()
        self.units = units
        self.type_encoder = type_encoder

        if type_encoder == 'lstm':
            self.encoder = nn.LSTM(input_size=units // 2, hidden_size=units, num_layers=2, bidirectional=False,
                                   batch_first=True,
                                   dropout=0)
            self.encoder2 = nn.ModuleList(
                nn.LSTM(input_size=units // 2, hidden_size=units, num_layers=2, bidirectional=False,
                        batch_first=True, dropout=0) for i in range(num_node))
        elif type_encoder == 'gru':
            self.encoder = nn.GRU(input_size=units // 2, hidden_size=units, num_layers=2, bidirectional=False,
                                  batch_first=True,
                                  dropout=0)
            self.encoder2 = nn.ModuleList(
                nn.GRU(input_size=units // 2, hidden_size=units, num_layers=2, bidirectional=False,
                       batch_first=True, dropout=0) for i in range(num_node))
        elif self.type_encoder == 'informer':
            self.encoder = InformerStack(enc_in=input_shape1, dec_in=input_shape1, c_out=units, seq_len=seq_len,
                                         label_len=1,
                                         out_len=units,
                                         factor=5, d_model=units, n_heads=6, e_layers=[3, 2, 1], d_layers=2,
                                         d_ff=512,
                                         dropout=0.0, attn='prob', embed='fixed', freq='d', activation='gelu',
                                         output_attention=False, distil=True, mix=True,
                                         device=device)
            self.encoder2 = nn.ModuleList(
                InformerStack(enc_in=input_shape1, dec_in=input_shape1, c_out=units, seq_len=seq_len, label_len=1,
                              out_len=units,
                              factor=5, d_model=units, n_heads=6, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                              dropout=0.0, attn='prob', embed='fixed', freq='d', activation='gelu',
                              output_attention=False, distil=True, mix=True,
                              device=device) for i in range(num_node))

        self.criterion = nn.SmoothL1Loss()

        # self.fusion = nn.Linear(num_node + 1, 1)
        # nn.init.xavier_uniform_(self.fusion.weight)

        self.mlp = nn.Sequential(
            nn.Linear(units * 4, units),
            # nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(units, 1)
        )
        for m in self.mlp.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        # self.end_conv = nn.Conv2d(1, 1, kernel_size=(1, self.units), bias=True)
        # nn.init.xavier_uniform_(self.end_conv.weight)

        self.emb = nn.Linear(input_shape1, units // 2)
        self.Q = nn.Linear(units, units)
        self.K = nn.Linear(units, units)
        self.V = nn.Linear(units, units)
        # self.M = nn.Linear(units, units)

        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.K.weight)
        # nn.init.xavier_uniform_(self.M.weight)

    def forward(self, ts, gts):
        if self.type_encoder == 'lstm':
            seq_a, (h_a, c_a) = self.encoder(self.emb(ts))
            seq_a, _ = attention_net(seq_a, seq_a)
            # seq_a = self.Q(seq_a)

            # print(seq_f.shape)
            p_seq = []
            p_h = []
            temp_h = []
            temp_seq = []
            # print(h_a.squeeze(0).shape)
            for idx in range(gts.shape[1]):
                seq_n, (h_n, c_n) = self.encoder2[idx](self.emb(gts[:, idx, :, :]))
                seq_n, _ = attention_net(seq_n, seq_n)
                # print(torch.mm(seq_a.unsqueeze(1), seq_n.unsqueeze(1).permute(0, 2, 1)).shape)
                p_seq.append(
                    torch.matmul(self.Q(seq_a).unsqueeze(1),
                                 self.K(seq_n).unsqueeze(1).permute(0, 2, 1)) / math.sqrt(
                        self.units))
                p_h.append(
                    torch.matmul(self.Q(h_a[-1, :, :]).unsqueeze(1),
                                 self.K(h_n[-1, :, :]).unsqueeze(1).permute(0, 2, 1)) / math.sqrt(
                        self.units))
                temp_seq.append(seq_n.unsqueeze(1))
                temp_h.append(h_n[-1, :, :].unsqueeze(1))
            p_seq = torch.cat(p_seq, dim=-1)
            p_seq = F.softmax(p_seq, dim=-1)
            p_h = torch.cat(p_h, dim=-1)
            p_h = F.softmax(p_h, dim=-1)
            # print(p)
            # print(torch.mm(p, torch.cat(temp, dim=0)).shape)
            # print(h_a[-1, :, :].squeeze(0).shape)

            ret = self.mlp(
                torch.cat(
                    [seq_a, h_a[-1, :, :], torch.matmul(p_h, self.V(torch.cat(temp_h, dim=1))).squeeze(1),
                     torch.matmul(p_seq, self.V(torch.cat(temp_seq, dim=1))).squeeze(1)],
                    dim=-1))

            # ret = F.sigmoid(ret)
        if self.type_encoder == 'gru':
            seq_a, h_a = self.encoder(self.emb(ts))
            seq_a, _ = attention_net(seq_a, seq_a)
            # seq_a = self.Q(seq_a)

            # print(seq_f.shape)
            p = []
            temp = []
            # print(h_a.squeeze(0).shape)
            for idx in range(gts.shape[1]):
                seq_n, h_n = self.encoder2[idx](self.emb(gts[:, idx, :, :]))
                seq_n, _ = attention_net(seq_n, seq_n)
                # print(torch.mm(seq_a.unsqueeze(1), seq_n.unsqueeze(1).permute(0, 2, 1)).shape)
                p.append(
                    torch.matmul(self.Q(seq_a).unsqueeze(1),
                                 self.K(seq_n).unsqueeze(1).permute(0, 2, 1)) / math.sqrt(
                        self.units))
                temp.append(self.M(h_n[-1, :, :]).unsqueeze(1))
            p = torch.cat(p, dim=-1)

            p = F.softmax(p, dim=-1)
            # print(p)
            # print(torch.mm(p, torch.cat(temp, dim=0)).shape)
            # print(h_a[-1, :, :].squeeze(0).shape)
            ret = self.mlp(
                torch.cat([self.M(h_a[-1, :, :]), torch.matmul(p, self.V(torch.cat(temp, dim=1))).squeeze(1)],
                          dim=-1))

            # ret = self.end_conv(
            # torch.cat([h_a[-1, :, :].unsqueeze(1), torch.matmul(p, self.V(torch.cat(temp, dim=1)))],
            # dim=1))


        elif self.type_encoder == 'informer':
            h_a = self.encoder(ts, ts)
            p = []
            temp = []
            for idx in range(gts.shape[1]):
                h_n = self.encoder2[idx](gts[:, idx, :, :], gts[:, idx, :, :])
                p.append(torch.matmul(h_a.unsqueeze(1), h_n.unsqueeze(1).permute(0, 2, 1)) / math.sqrt(self.units))
                temp.append(h_n.unsqueeze(1))
            p = torch.cat(p, dim=-1)
            p = F.softmax(p, dim=-1)

            ret = self.mlp(
                torch.cat([h_a, torch.matmul(p, torch.cat(temp, dim=1)).squeeze(1)], dim=-1))
            # ret = F.sigmoid(ret)

        # print(ret.shape)
        return ret


class detailed_attention_ts(nn.Module):
    def __init__(self, units=64, num_node=2, input_shape1=None, seq_len=11, device="cuda:0", type_encoder='lstm'):
        super(detailed_attention_ts, self).__init__()
        self.units = units
        self.type_encoder = type_encoder
        self.seq_len = seq_len
        self.num_node = num_node
        self.device = device
        self.ATT = ATT(units)

        if type_encoder == 'lstm':
            self.encoder = nn.ModuleList(
                nn.LSTMCell(input_size=units // 2, hidden_size=units, bias=True) for i in range(seq_len))
            self.encoder2 = nn.ModuleList(
                nn.ModuleList(
                    nn.LSTMCell(input_size=units // 2, hidden_size=units, bias=True) for i in range(seq_len)) for i
                in
                range(num_node))
        elif type_encoder == 'gru':
            self.encoder = nn.ModuleList(
                nn.GRUCell(input_size=units // 2, hidden_size=units, bias=True) for i in range(seq_len))
            self.encoder2 = nn.ModuleList(
                nn.ModuleList(
                    nn.GRUCell(input_size=units // 2, hidden_size=units, bias=True) for i in range(seq_len)) for i
                in
                range(num_node))

        self.criterion = nn.SmoothL1Loss()

        self.mlp = nn.Sequential(
            nn.Linear(units, units // 2),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(units // 2, 1)
        )
        for m in self.mlp.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        # self.end_conv = nn.Conv2d(1, 1, kernel_size=(1, self.units), bias=True)
        # nn.init.xavier_uniform_(self.end_conv.weight)

        self.norm = nn.BatchNorm1d(units, affine=False)

        self.emb = nn.Linear(input_shape1, units // 2)
        self.Q = nn.Linear(units, units)
        self.K = nn.Linear(units, units)
        self.V = nn.Linear(units, units)
        self.fuse = nn.Linear(units * 2, units)
        self.fuse1 = nn.Linear(units * 2, units)
        self.fuse2 = nn.Linear(units * 2, units)
        # self.M = nn.Linear(units, units)

        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.fuse1.weight)
        nn.init.xavier_uniform_(self.fuse2.weight)
        # nn.init.xavier_uniform_(self.M.weight)
    def att_mat(self, ts, gts):
        temp = torch.cat((ts.unsqueeze(1), gts), dim=1)


        h_ts, _ = self.encoder[0](self.emb(ts))
        h_gts = []
        for idx in range(gts.shape[1]):
            temp, _ = self.encoder2[idx][0](self.emb(gts[:, idx, :]))
            h_gts.append(temp.unsqueeze(1))
        # keys = torch.cat((h_ts.unsqueeze(1), torch.cat(h_gts, dim=1)), dim=1)[0, :, :]
        keys = torch.cat(h_gts, dim=1)[0, :, :]
        # query = keys[0, :].unsqueeze(0)
        query = h_ts[0, :].unsqueeze(0)
        p = []
        for k in range(keys.shape[0]):
            p.append(torch.matmul(self.ATT.Q(query),
                                  self.ATT.K(keys[k, :]).unsqueeze(0).T) / math.sqrt(
                self.units))
        p = torch.cat(p, dim=-1)
        # mask = torch.tensor([[0, 0, 1]]).to(self.device).bool()
        # p = p.masked_fill_(mask, -float('inf'))
        # _mask = torch.tensor([[1, 0, 0]]).to(self.device).bool()

        # p = p.masked_fill_(_mask, float('inf'))
        # p = F.softmax(p, dim=-1)
        # return F.relu(self.ATT.fuse(torch.cat([torch.matmul(p, self.V(keys)).squeeze(1), query], dim=-1)))
        return p
        # h_ts_a = self.ATT(query, key, norm=False)

        # # a = torch.matmul(self.Q(ts), self.K(key).T) / math.sqrt(self.units)
        # a[0, 2] = 0
        # temp = F.softmax(a)



    def forward(self, ts, gts, distance=None):
        if distance is not None:
            distance = F.softmax(distance)
        if self.type_encoder == 'lstm':
            # self.att_mat(ts[:, 0, :], gts[:, :, 0, :])
            h_ts, _ = self.encoder[0](self.emb(ts[:, 0, :]))
            h_gts = []
            for idx in range(gts.shape[1]):
                temp, _ = self.encoder2[idx][0](self.emb(gts[:, idx, 0, :]))
                h_gts.append(temp.unsqueeze(1))

            h_ts_a = self.ATT(h_ts, torch.cat(h_gts, dim=1), norm=False)
            if distance is not None:
                h_ts_d = torch.matmul(distance.unsqueeze(1), torch.cat(h_gts, dim=1))
                h_ts_a = F.relu(self.fuse2(torch.cat([h_ts_d.squeeze(1), h_ts_a], dim=-1)))
            for i in range(len(h_gts)):
                h_gts[i] = F.relu(self.fuse1(torch.cat([h_gts[i].squeeze(1), h_ts], dim=-1)))

            c_ts = torch.zeros_like(h_ts)
            c_gts = [torch.zeros_like(h_ts) for j in range(gts.shape[1])]

            for i in range(1, self.seq_len):
                h_ts, c_ts = self.encoder[i](self.emb(ts[:, i, :]), (h_ts_a, c_ts))
                for j in range(gts.shape[1]):
                    temp, c_gts[j] = self.encoder2[j][i](self.emb(gts[:, j, i, :]), (h_gts[j], c_gts[j]))
                    h_gts[j] = temp.unsqueeze(1)
                h_ts_a = self.ATT(h_ts, torch.cat(h_gts, dim=1), norm=False)
                for idx in range(len(h_gts)):
                    h_gts[idx] = F.relu(self.fuse1(torch.cat([h_gts[idx].squeeze(1), h_ts], dim=-1)))

            ret = self.mlp(h_ts_a)

        if self.type_encoder == 'gru':
            h_ts = self.encoder[0](self.emb(ts[:, 0, :]))
            h_gts = []
            for idx in range(gts.shape[1]):
                temp = self.encoder2[idx][0](self.emb(gts[:, idx, 0, :]))
                h_gts.append(temp.unsqueeze(1))
            h_ts_a = self.ATT(h_ts, torch.cat(h_gts, dim=1), norm=False)
            if distance is not None:
                h_ts_d = torch.matmul(distance.unsqueeze(1), torch.cat(h_gts, dim=1))
                h_ts_a = F.relu(self.fuse2(torch.cat([h_ts_d.squeeze(1), h_ts_a], dim=-1)))
            for i in range(len(h_gts)):
                h_gts[i] = F.relu(self.fuse1(torch.cat([h_gts[i].squeeze(1), h_ts], dim=-1)))

            for i in range(1, self.seq_len):
                h_ts = self.encoder[i](self.emb(ts[:, i, :]), h_ts_a)
                for j in range(gts.shape[1]):
                    temp = self.encoder2[j][i](self.emb(gts[:, j, i, :]), h_gts[j])
                    h_gts[j] = temp.unsqueeze(1)
                h_ts_a = self.ATT(h_ts, torch.cat(h_gts, dim=1), norm=False)
                for idx in range(len(h_gts)):
                    h_gts[idx] = F.relu(self.fuse1(torch.cat([h_gts[idx].squeeze(1), h_ts], dim=-1)))

            ret = self.mlp(h_ts_a)

        return ret


class single_ts(nn.Module):
    def __init__(self, units=64, input_shape1=None, seq_len=11, device="cuda:0", type_encoder='lstm'):
        super(single_ts, self).__init__()
        self.units = units
        self.type_encoder = type_encoder
        if self.type_encoder == 'lstm':
            self.embed = nn.Linear(input_shape1, units, bias=True)
            self.encoder = nn.LSTM(input_size=input_shape1, hidden_size=units, num_layers=1, bidirectional=False,
                                   batch_first=True,
                                   dropout=0.2)
        elif self.type_encoder == 'gru':
            self.embed = nn.Linear(input_shape1, units, bias=True)
            self.encoder = nn.GRU(input_size=input_shape1, hidden_size=units, num_layers=1, bidirectional=False,
                                  batch_first=True,
                                  dropout=0.2)
        elif self.type_encoder == 'informer':
            self.encoder = InformerStack(enc_in=input_shape1, dec_in=input_shape1, c_out=1, seq_len=seq_len,
                                         label_len=1,
                                         out_len=1,
                                         factor=5, d_model=units, n_heads=6, e_layers=[3, 2, 1], d_layers=2,
                                         d_ff=512,
                                         dropout=0.0, attn='prob', embed='fixed', freq='d', activation='gelu',
                                         output_attention=False, distil=True, mix=True,
                                         device=device)

        self.criterion = nn.SmoothL1Loss()

        self.mlp = nn.Sequential(
            # nn.Linear(units, units // 2),
            # nn.Dropout(p=0.2),
            # nn.ReLU(),
            nn.Linear(units, 1)
        )
        for m in self.mlp.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, ts, gts):
        if self.type_encoder == 'lstm':
            seq_a, (h_a, c_a) = self.encoder(ts)
            ret = self.mlp(h_a.squeeze(0))  # F.sigmoid(self.mlp(h_a.squeeze(0)))
            return ret
        elif self.type_encoder == 'gru':
            seq_a, h_a = self.encoder(ts)
            ret = self.mlp(h_a.squeeze(0))  # F.sigmoid(self.mlp(h_a.squeeze(0)))
            return ret
        elif self.type_encoder == 'informer':
            # print(1)
            h_a = self.encoder(ts, ts)
            ret = h_a  # F.sigmoid(h_a)
            return ret


class RMGM(nn.Module):
    def __init__(self, num_nodes, input_shape1, units, seq_len, polynomials, L_tilde,
                 num_layers=2, output_dim=1, rnn_units=64, cheb_k=2, device='cpu'):
        super(RMGM, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_shape1
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.device = device

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, units), requires_grad=True)

        self.encoder = RGCN(polynomials, L_tilde, num_nodes, input_shape1, rnn_units, cheb_k,
                            units, num_layers)

        # predictor
        self.end_conv = nn.Conv2d(1, seq_len * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def scaled_laplacian(self, node_embeddings, is_eval=False):
        # Normalized graph Laplacian function.
        # :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
        # :return: np.matrix, [n_route, n_route].
        # learned graph
        node_num = self.num_node
        learned_graph = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
        norm = torch.norm(node_embeddings, p=2, dim=1, keepdim=True)
        norm = torch.mm(norm, norm.transpose(0, 1))
        learned_graph = learned_graph / norm
        learned_graph = (learned_graph + 1) / 2.
        # learned_graph = F.sigmoid(learned_graph)
        learned_graph = torch.stack([learned_graph, 1 - learned_graph], dim=-1)

        # make the adj sparse
        if is_eval:
            adj = F.gumbel_softmax(learned_graph, tau=1, hard=True)
        else:
            adj = F.gumbel_softmax(learned_graph, tau=1, hard=True)
        adj = adj[:, :, 0].clone().reshape(node_num, -1)
        # mask = torch.eye(self.num_nodes, self.num_nodes).to(device).byte()
        mask = torch.eye(node_num, node_num).bool().to(self.device)
        adj.masked_fill_(mask, 0)

        # d ->  diagonal degree matrix
        W = adj
        n = W.shape[0]
        d = torch.sum(W, axis=1)
        ## L -> graph Laplacian
        L = -W
        L[range(len(L)), range(len(L))] = d
        try:

            lambda_max = (L.max() - L.min())
        except Exception as e:
            print("eig error!!: {}".format(e))
            lambda_max = 1.0

        tilde = (2 * L / lambda_max - torch.eye(n).to(self.device))
        self.adj = adj
        self.tilde = tilde
        return adj, tilde

    def forward(self, source=None, gts=None):
        if self.train:
            adj, learned_tilde = self.scaled_laplacian(self.node_embeddings, is_eval=False)
        else:
            adj, learned_tilde = self.scaled_laplacian(self.node_embeddings, is_eval=True)
        gts = gts.permute(0, 2, 1, 3)
        init_state = self.encoder.init_hidden(gts.shape[0])
        output, _ = self.encoder(gts, init_state, self.node_embeddings, learned_tilde)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output