import sys
import os
import math
import pandas as pd

from preprocess_utils.data_processing import Data_loader, split_kge_ts_input

sys.path.append(os.path.join(sys.path[0], '..'))
from models import *

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
from torch.utils import data
import random
from preprocess_utils.gpu_mem_track import MemTracker

torch.manual_seed(102)
torch.cuda.manual_seed_all(102)
np.random.seed(102)
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataset_util import kge_ts_dataset, kge_ts_traffic_dataset
from functions import model_train, model_test

import time
from preprocess_utils.logger import get_logger
import argparse

if __name__ == '__main__':
    gpu_tracker = MemTracker()
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument("--num_node", default=2,  type=int)
    args.add_argument("--hop", default=1, type=int)
    args.add_argument("--time_window", default=16, type=int)
    args = args.parse_args()
    print(args)
    close_num_node = args.num_node
    hop = args.hop
    time_window = args.time_window

    with_full_graph = False
    with_distance = False
    lr_init = 0.0001
    lr_decay = False
    lr_decay_rate = 0.3
    lr_decay_step = 5, 20, 40, 70
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_name = 'PEMS03'
    save_name = time.strftime("%m-%d-%Hh%Mm") + "_" + data_name + "_" + "window{" + str(time_window) + "}" + "_hop{" + str(hop) + "}" + "_num_node{" + str(close_num_node) + "}"
    path = "./runs"
    log_dir = os.path.join(path, data_name, save_name)
    logger = get_logger(root=log_dir, name=data_name, debug=False)

    test = Data_loader()
    test.read_data(num_node=close_num_node, start_index=0, end_index=-1, data_name=data_name, with_graph=with_full_graph, hop=hop)
    # subset attention score compute
    #180-183,324-183
    # test.read_data(num_node=close_num_node, index_list=[324, 180, 183], data_name='PEMS03',with_graph=with_full_graph, hop=hop)
    # #129-130， 130-107， 107-134
    # test.read_data(num_node=close_num_node, index_list=[134, 107, 129, 130], data_name='PeMSD4', with_graph=with_full_graph, hop=hop)

    x_set, y_set = test.data_processing_kge_ts_input(time_window=time_window)
    # train_x, valid_x, train_y, valid_y = train_test_split(x_set, y_set, test_size=0.2,
    #                                                       random_state=102)
    train_x = x_set
    train_y = y_set

    if with_distance:
        x_price, x_gts, x_distance = split_kge_ts_input(train_x, with_distance)
    else:
        x_price, x_gts = split_kge_ts_input(train_x, with_distance)

    if with_full_graph:
        test.cheb_polynomials = [i.to(device) for i in test.cheb_polynomials]
        test.adj_mx = test.adj_mx.to(device)
        test.L_tilde = test.adj_mx.to(device)
        tm_model = RMGM(num_nodes=test.num_of_vertices, input_shape1=3, units=64, seq_len=time_window,
                        polynomials=test.cheb_polynomials, L_tilde=test.L_tilde, device=device)
    else:
        tm_model = detailed_attention_ts(units=64, input_shape1=3, num_node=close_num_node, device=device,
                                         seq_len=time_window,
                                         type_encoder='lstm')
    gpu_tracker.track()
    tm_model = tm_model.to(device)
    gpu_tracker.track()
    optimizer = torch.optim.Adam(params=tm_model.parameters(), lr=lr_init, eps=1.0e-8,
                                 weight_decay=0, amsgrad=False)
    # learning rate decay
    lr_scheduler = None
    if lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=lr_decay_rate)
    if with_distance:
        train_data = kge_ts_traffic_dataset(x_price, x_gts, x_distance, y=train_y)
    else:
        train_data = kge_ts_dataset(x_price, x_gts, y=train_y)

    train_iter = data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=False,
        num_workers=4)

    # if with_distance:
    #     x_price, x_gts, x_distance = split_kge_ts_input(valid_x, with_distance)
    #     valid_data = kge_ts_traffic_dataset(x_price, x_gts, x_distance, y=valid_y)
    # else:
    #     x_price, x_gts = split_kge_ts_input(valid_x, with_distance)
    #     valid_data = kge_ts_dataset(x_price, x_gts, valid_y)

    # valid_iter = data.DataLoader(
    #     dataset=valid_data,
    #     batch_size=64,
    #     shuffle=False,
    #     num_workers=4)
    gpu_tracker.track()
    for epoch in range(1, 20):
        pre_loss = model_train(epoch, tm_model, train_iter, optimizer, device, with_distance, logger)
        if epoch == 4:
            break
        gpu_tracker.track()
        # if epoch % 2 == 0:
        #     print()
        #     print('Valid result: ')
        #     model_test(tm_model, valid_iter, device, with_distance, logger)
        #     print()

    # x_price, x_gts = split_kge_ts_input(train_x)
    # torch.save(tm_model.state_dict(), 'kg_e_ts_051022.pth')
    # torch.save(tm_model.state_dict(), 'kg_e_ts_new.pth')
