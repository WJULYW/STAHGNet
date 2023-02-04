import warnings

warnings.filterwarnings('ignore')
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from torch import optim
import torch
import torch.nn as nn
import random

torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)
np.random.seed(1024)
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm


def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / (actual+1))) * 100



def model_train(epoch, model, train_iter, optimizer, device, with_distance, logger):
    model.train()
    num, loss_ = 0, 0.0
    y_pred = []
    y_true = []

    for i, batch in tqdm(enumerate(train_iter)):
        all_ts = train_iter.dataset.ts
        all_gts = train_iter.dataset.gts
        optimizer.zero_grad()
        num += 1
        if not with_distance:
            ts, gts, y = batch
            ts = ts.to(device)
            gts = gts.to(device)
            ts = ts.float()
            gts = gts.float()
            y = y.float()
            y = y.to(device)
            res = model(ts, gts)
            # if epoch == 4 and i == 0:
            #     for j in [0, 2, 4, 6, 8, 10]:
            #         print("epoch: " + str(epoch) + " " + str(model.att_mat(ts[:, j, :], gts[:, :, j, :])))
            #     break
        else:
            ts, gts, distance, y = batch
            ts = ts.to(device)
            gts = gts.to(device)
            distance = distance.to(device)
            ts = ts.float()
            gts = gts.float()
            distance = distance.float()
            y = y.float()
            y = y.to(device)
            # if epoch == 4 and i == 0:
            #     for j in [0, 2, 4, 6, 8, 10]:
            #         print("epoch: " + str(epoch) + " " + str(model.att_mat(ts[:, j, :], gts[:, :, j, :])))
            #     break

            res = model(ts, gts, distance)

        loss = model.criterion(res.squeeze(-1), y)

        loss_ += loss.item()
        y = y.cpu().numpy()
        y_true += y.tolist()
        res = res.squeeze().detach().cpu().numpy().tolist()
        # print(res.shape)
        # print(res)
        y_pred += res
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()
    # logger.info("epoch: " + str(epoch) + " average matching loss: " + str(loss_ / num))
    # logger.info("MAE: " + str(mean_absolute_error(y_true, y_pred)))
    # logger.info("RMSE: " + str(mean_squared_error(y_true, y_pred, squared=False)))
    # logger.info("MAPE: " + str(mean_absolute_percentage_error(y_true, y_pred)))
    #
    # print("epoch: " + str(epoch) + " average matching loss: " + str(loss_ / num))
    # print("MAE: ", mean_absolute_error(y_true, y_pred))
    # print("RMSE: ", mean_squared_error(y_true, y_pred, squared=False))
    # print("MAPE: ", mape(y_true, y_pred))
    return loss_ / num


def model_test(model, test_iter, device, with_distance, logger):
    model.eval()
    y_pred = []
    y_true = []
    for i, batch in tqdm(enumerate(test_iter)):
        if not with_distance:
            ts, gts, y = batch
            ts = ts.to(device)
            gts = gts.to(device)
            ts = ts.float()
            gts = gts.float()
            y = y.float()
            res = model(ts, gts)
        else:
            ts, gts, distance, y = batch
            ts = ts.to(device)
            gts = gts.to(device)
            distance = distance.to(device)
            ts = ts.float()
            gts = gts.float()
            distance = distance.float()
            y = y.float()
            res = model(ts, gts, distance)

        y = y.numpy()
        y_true += y.tolist()
        res = model(ts, gts)
        # print(res)

        res = res.squeeze().detach().cpu().numpy().tolist()

        y_pred += res
    #print(y_true[:10])
    #print(y_pred[:10])
    logger.info("MAE: " + str(mean_absolute_error(y_true, y_pred)))
    logger.info("RMSE: " + str(mean_squared_error(y_true, y_pred, squared=False)))
    logger.info("MAPE: " + str(mean_absolute_percentage_error(y_true, y_pred)))
    print("MAE: ", mean_absolute_error(y_true, y_pred))
    print("RMSE: ", mean_squared_error(y_true, y_pred, squared=False))
    print("MAPE: ", mape(y_true, y_pred))
