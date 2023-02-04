import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
import random
import math

torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)
np.random.seed(1024)
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_size, device, output_size, seq_len, nhead, num_layers=1, dropout=0):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        # self.d_model = d_model
        self.device = device
        self.output_size = output_size
        self.seq_len = seq_len
        self.nhead = nhead
        self.dropout = dropout

        self.input_fc = nn.Linear(input_size, output_size)

        self.pos_emb = PositionalEncoding(self.output_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_size,
            nhead=self.nhead,
            dim_feedforward=self.nhead * self.output_size,
            batch_first=True,
            dropout=self.dropout,
            device=self.device
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.output_size,
            nhead=self.nhead,
            dropout=self.dropout,
            dim_feedforward=self.nhead * self.output_size,
            batch_first=True,
            device=self.device
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # self.fc = nn.Linear(self.output_size * self.input_size, self.output_size)
        self.fc1 = nn.Linear(self.seq_len * self.output_size, 1)
        # self.fc2 = nn.Linear(self.output_size, self.output_size)

    def forward(self, x):
        # print(x.shape)
        x = self.pos_emb(self.input_fc(x))  # (256, 24, 128)
        x = self.encoder(x)
        # print(x.shape)
        # 不经过解码器
        x = x.flatten(start_dim=1)
        # x = self.fc1(x)
        out = self.fc1(x)
        # y = self.output_fc(y)   # (256, 4, 128)
        # out = self.decoder(y, x)  # (256, 4, 128)
        # out = out.flatten(start_dim=1)  # (256, 4 * 128)
        # out = self.fc(out)  # (256, 4)

        return out
