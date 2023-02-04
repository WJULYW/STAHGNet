import numpy as np
from torch.utils.data import Dataset
from typing import Sequence, Optional


class kge_ts_dataset(Dataset):
    def __init__(self, ts, gts, y=None, isTest=False):
        self.ts = ts
        self.gts = gts
        self.y = y
        self.isTest = isTest

    def __getitem__(self, idx):
        if self.isTest:
            return self.ts[idx], self.gts[idx]
        else:
            return self.ts[idx], self.gts[idx], self.y[idx]

    def __len__(self):
        return len(self.ts)


class kge_ts_traffic_dataset(Dataset):
    def __init__(self, ts, gts, dis, y=None, isTest=False):
        self.ts = ts
        self.gts = gts
        self.dis = dis
        self.y = y
        self.isTest = isTest

    def __getitem__(self, idx):
        if self.isTest:
            return self.ts[idx], self.gts[idx], self.dis[idx]
        else:
            return self.ts[idx], self.gts[idx], self.dis[idx], self.y[idx]

    def __len__(self):
        return len(self.ts)
