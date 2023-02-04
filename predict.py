import sys
import os
import math
import pandas as pd

sys.path.append(os.path.join(sys.path[0], '..'))
from data_utils.data_utils import DataLoader
from data_processing import Stock_loader, split_kge_ts_input
from models import *

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
from torch.utils import data
import random

torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)
np.random.seed(1024)
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataset_util import kge_ts_dataset
from functions import model_train, model_test
from backtest_strategy.data_preprocessing.Data_processing_backtest import data_processing
from backtest_strategy.utils.backtest_strategy import backtesting

if __name__ == '__main__':
    PATH = 'kg_e_ts_new.pth'
    test = Stock_loader()
    num_node = 2
    time_window = 11
    test.choose_stock(stock_list=['000078.SZ', '600053.SH', '600061.SH','600640.SH'], kg_path='final_uie.csv', num_node=num_node,
                      start_date=20190401, end_date=20200610)


    x_set, y_set, date_set, code_set = test.data_processing_kge_ts_input(time_window=time_window, mode='test')
    x_price, x_gts = split_kge_ts_input(x_set)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tm_model = attention_ts(units=64, input_shape1=8, device=device, seq_len=time_window, type_encoder='lstm')
    tm_model.load_state_dict(torch.load(PATH))
    tm_model = tm_model.to(device)
    test_data = kge_ts_dataset(x_price, x_gts, isTest=True)
    test_iter = data.DataLoader(
        dataset=test_data,
        batch_size=16,
        shuffle=False,
        num_workers=1)
    tm_model.eval()
    y_pred = []
    for i, batch in tqdm(enumerate(test_iter)):
        ts, gts = batch
        ts = ts.to(device)
        gts = gts.to(device)
        ts = ts.float()
        gts = gts.float()
        res = tm_model(ts, gts)
        res = [1 if y >= 0.5 else 0 for y in res.detach().cpu().numpy()]
        # print(res)

        y_pred += res
    print(sum(y_pred))
    '''
    dicct = {'date': date_set, 'code': code_set, 'signal': y_pred}
    res = pd.DataFrame(dicct)
    print(sum(y_pred))
    res.to_csv('test_res_kge.csv', index=False)'''


    '''
    model = tf.keras.models.load_model('test_only_ts_model')

    signal = model.predict(x_set)
    signal = [x[0] for x in signal]
    dicct = {'date': date_set, 'code': code_set, 'signal': signal}
    res = pd.DataFrame(dicct)
    print(res.head(10))
    res.to_csv('test_res_only_ts.csv', index=False)'''
