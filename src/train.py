# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, itertools, random, argparse, time, datetime
import numpy as np
import random

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
from math import sqrt

import scipy.sparse as sp
from scipy.stats.stats import pearsonr
from models import *
from data import *

import shutil
import logging
import glob
import time
import scipy.fft as fft
import dtw
from sklearn.preprocessing import MinMaxScaler
import csv
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

# Training settings
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='state360', help="Dataset string")
ap.add_argument('--sim_mat', type=str, default='region-adj', help="adjacency matrix filename (*-adj.txt)")
ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)") 
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
ap.add_argument('--batch', type=int, default=100, help="batch size")
ap.add_argument('--check_point', type=int, default=1, help="check point")
ap.add_argument('--shuffle', action='store_true', default=False, help="not used, default false")
ap.add_argument('--train', type=float, default=.6, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.2, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.2, help="Testing ratio (0, 1)")
ap.add_argument('--model', default='cola_gnn', choices=['cola_gnn','CNNRNN_Res','RNN','AR','ARMA','VAR','GAR','SelfAttnRNN','lstnet','stgcn','dcrnn'], help='')
ap.add_argument('--rnn_model', default='RNN', choices=['LSTM','RNN','GRU'], help='')
ap.add_argument('--mylog', action='store_false', default=True,  help='save tensorboad log')
ap.add_argument('--cuda', action='store_true', default=True,  help='')
ap.add_argument('--window', type=int, default=20, help='')
ap.add_argument('--horizon', type=int, default=15, help='leadtime default 1')
ap.add_argument('--save_dir', type=str,  default='save',help='dir path to save the final model')
ap.add_argument('--gpu', type=int, default=0,  help='choose gpu 0-10')
ap.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
ap.add_argument('--bi', action='store_true', default=False,  help='bidirectional default false')
ap.add_argument('--patience', type=int, default=500, help='patience default 100')
ap.add_argument('--k', type=int, default=10,  help='kernels')
ap.add_argument('--hidsp', type=int, default=15,  help='spatial dim')

args = ap.parse_args() 
print('--------Parameters--------')
print(args)
print('--------------------------')

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dcrnn_model import *


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda:0")
args.cuda = args.cuda and torch.cuda.is_available() 
logger.info('cuda %s', args.cuda)

time_token = str(time.time()).split('.')[0] # tensorboard model
log_token = '%s.%s.w-%s.h-%s.%s' % (args.model, args.dataset, args.window, args.horizon, args.rnn_model)
"""
if args.mylog:
    tensorboard_log_dir = 'tensorboard/%s' % (log_token)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    shutil.rmtree(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)
"""
#data_loader = DataBasicLoader(args)

"""
if args.model == 'CNNRNN_Res':
    model = CNNRNN_Res(args, data_loader)  
elif args.model == 'RNN':
    model = RNN(args, data_loader)
elif args.model == 'AR':
    model = AR(args, data_loader)
elif args.model == 'ARMA':
    model = ARMA(args, data_loader)
elif args.model == 'VAR':
    model = VAR(args, data_loader)
elif args.model == 'GAR':
    model = GAR(args, data_loader)
elif args.model == 'SelfAttnRNN':
    model = SelfAttnRNN(args, data_loader)
elif args.model == 'lstnet':
    model = LSTNet(args, data_loader)      
elif args.model == 'stgcn':
    model = STGCN(args, data_loader, data_loader.m, 1, args.window, 1)  
elif args.model == 'dcrnn':
    model = DCRNNModel(args, data_loader)   
elif args.model == 'cola_gnn':
    model = cola_gnn(args, data_loader)        
else: 
    raise LookupError('can not find the model')
 
logger.info('model %s', model)
if args.cuda:
    model.cuda()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params:',pytorch_total_params)
"""
def graph_module(X):
    x_temp = X[0]
    # 배치 전체를 하나의 matrix로 생성
    for i in range(len(X)):
        if i == 0:
            pass
        else:
            x = X[i]
            # x_last = torch.Tensor([x[-1]])
            x_last = torch.unsqueeze(x[-1], dim=0)
            x_temp = torch.cat([x_temp, x_last], dim=0)
    # print("xtemp :", x_temp)
    # 정규화
    scaler = MinMaxScaler()
    x_temp = scaler.fit_transform((x_temp.cpu().numpy()))
    # (5000,8)을 (8,5000)으로 변경
    x_temp = x_temp.transpose(1, 0)

    # x_temp_pd = pd.DataFrame(x_temp)
    # x_temp_pd.to_csv("./pdseries.csv")

    for i in range(len(x_temp)):
        globals()['x_{}'.format(i)] = x_temp[i]
    # 푸리에변환으로 frequency 분해
    for i in range(len(x_temp)):
        fft_temp = fft.fft(globals()['x_{}'.format(i)])[:30]
        temp_list = []
        for j in range(10):
            temp = str(fft_temp[j])
            #print(temp)
            try:
                temp = float(temp[1:5])
            except:
                try:
                    temp = float(temp[1])
                except:
                    temp = float(temp[2])
            temp_list.append(temp)

        globals()['x_{}_list'.format(i)] = temp_list

    # 두 리스트의 원소곱으로 유사도 비교
    similarity = np.zeros(shape=(len(x_temp), len(x_temp)))
    for i in range(len(x_temp)):
        for j in range(i, len(x_temp)):
            similarity_temp = np.sum(np.multiply(globals()['x_{}_list'.format(i)][1:], \
                                                 globals()['x_{}_list'.format(j)][1:]))
            # print("a : ",globals()['x_{}_list'.format(i)])
            # print("b : ", globals()['x_{}_list'.format(j)])
            similarity[i][j] = similarity_temp
            similarity[j][i] = similarity_temp
            # print("temp : ", similarity_temp)
    similarity = np.round_(similarity, 2)

    for i in range(len(similarity)):
        for j in range(len(similarity)):
            if i == j:
                similarity[i][j] = 0

    similarity_temp = pd.DataFrame(similarity)
    similarity_temp.to_csv("./similarity3.csv")
    # 정규분포화
    similarity_std = (similarity - np.mean(similarity)) / np.std(similarity)
    similarity_std = np.where(similarity_std > 0, similarity_std, 0)
    # 정규분포 상 0보다 큰 애들의 index에 대해서만 similarity 원래 값을 가져옴
    similarity = np.round_(np.where(similarity_std > 0, similarity, 0), 2)
    # print("origin : \n", similarity.astype(int))
    # 선후관계 파악을 위한 DTW
    index_list = []
    for i in range(len(similarity)):
        for j in range(i, len(similarity)):
            if similarity[i][j] != 0:
                index_list.append([i, j])

    for relation in index_list:
        # 고정시키는게 i, 움직이는게 j
        relation1 = globals()['x_{}'.format(relation[0])]
        relation2 = globals()['x_{}'.format(relation[1])]

        relation1_startindex = len(relation1) // 10
        relation1_mid = relation1[relation1_startindex:-relation1_startindex]

        moving = (len(relation1) - len(relation1_mid)) // 10
        start_idx = 0
        min_dtw = np.inf
        min_index = np.nan
        for i in range(10):
            relation2_temp = relation2[start_idx:start_idx + len(relation1_mid)]
            dtw_temp = dtw.dtw(relation2_temp, relation1_mid, keep_internals=True).distance
            if min_dtw > dtw_temp:
                min_dtw = dtw_temp
                min_index = i
            start_idx += moving
        if min_index < 5:
            # relation2가 앞서있는거 -> relation2는 그대로, relation1에 대해서 penalty
            # penalty는 min index값에 대해서 linear하게 설정
            similarity[relation[0]][relation[1]] /= (((np.abs(4.5 - min_index)) / 9) + 1)
        else:
            # relation1이 앞서있는거 -> relation1은 그대로, relation2에 대해서 penalty
            similarity[relation[1]][relation[0]] /= (((np.abs(4.5 - min_index)) / 9) + 1)
    # print("after : \n", similarity.astype(int))
    similarity = similarity.astype(int)
    max_value = np.max(similarity)
    similarity = np.round_(similarity / max_value, 4)
    # print("similarity :\n", similarity)

    # 허브노드의 연결 계산
    node_neighbors = []
    # adj = np.identity(len(similarity)) + similarity
    adj = similarity
    for i in range(len(similarity)):
        temp = 0
        for j in range(len(similarity)):
            temp += (np.log(adj[i][j] + 1))
        node_neighbors.append(temp)
    M = np.max(node_neighbors)
    m = np.min(node_neighbors)

    node_neighbors = [(1 / (1 + np.exp(-x + (m + M) / 2))) * \
                      (1 - np.log(1 + m / M)) + np.log(1 + m / M) for x in node_neighbors]
    node_neighbors = [x * (1 / max(node_neighbors)) for x in node_neighbors]

    for i in range(len(node_neighbors)):
        for j in range(len(node_neighbors)):
            adj[i][j] *= node_neighbors[j]
    adj += np.identity(len(adj))
    #adj = torch.Tensor(adj).to(device)
    return adj

def evaluate(data_loader, data, graph, tag='val'):
    model.eval()
    total = 0.
    n_samples = 0.
    total_loss = 0.
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        #graph = graph_module(X)

        output,_  = model(X, graph)
        loss_train = F.l1_loss(output, Y) # mse_loss
        total_loss += loss_train.item()
        n_samples += (output.size(0) * data_loader.m);

        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx) # [n_samples, 47] 
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  #(#n_samples, 47)
    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values'))) # mean of 47
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae) # Standard deviation of MAEs for all states/places 
    pcc_tmp = []
    for k in range(data_loader.m):
        pcc_tmp.append(pearsonr(y_true_states[:,k],y_pred_states[:,k])[0])
    pcc_states = np.mean(np.array(pcc_tmp)) 
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    # convert y_true & y_pred to real data
    y_true = np.reshape(y_true_states,(-1))
    y_pred = np.reshape(y_pred_states,(-1))
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    #print("Y_true :", y_true.shape)
    #print("Y_true2 :", y_pred.shape)
    pcc = pearsonr(y_true,y_pred)[0]
    r2 = r2_score(y_true, y_pred,multioutput='uniform_average') #variance_weighted 
    var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)
    global y_true_t
    global y_pred_t
    y_true_t = y_true_states
    y_pred_t = y_pred_states
    return float(total_loss / n_samples), mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae

def train(data_loader, data):
    #print(data[0].shape[2])
    data_len = data[0].shape[2]
    model.train()
    total_loss = 0.
    n_samples = 0.
    batch_size = args.batch
    graph = np.zeros(data_len)
    idx = 0
    for inputs in data_loader.get_batches(data, batch_size, True):
        idx+=1
        X, Y = inputs[0], inputs[1]
        graph_temp = graph_module(X)
        graph = graph_temp*(1/idx) + graph*((idx-1)/idx)
        graph_send = torch.Tensor(graph).to(device)
        optimizer.zero_grad()
        output, _  = model(X, graph_send)
        if Y.size(0) == 1:
            Y = Y.view(-1)
        loss_train = F.l1_loss(output, Y) # mse_loss
        total_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
        n_samples += (output.size(0) * data_loader.m)
    return float(total_loss / n_samples), graph_send

bad_counter = 0
best_epoch = 0
best_val = 1e+20;
device = torch.device('cuda:0')

dataset_list = ['exchange_rate']
horizon_list = [15]

for dataset in dataset_list:
    for horizon in horizon_list:
        args.dataset = dataset
        args.horizon = horizon
        data_loader = DataBasicLoader(args)
        model = cola_gnn(args, data_loader)
        model = model.to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('#params:', pytorch_total_params)
        try:
            print('begin training');
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            rmse_list = []
            pcc_list = []
            rmse_indexlist = []
            pcc_indexlist = []
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                train_loss, graph = train(data_loader, data_loader.train)

                val_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.val, graph)
                if epoch % 10 == 0:
                    print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))
                    print('rmse : {} / pcc : {}'.format(rmse, pcc))
                    print('--------------------')
                    print('--------------------')
                    test_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae = evaluate(
                        data_loader, data_loader.test, graph, tag='test')
                    rmse_list.append(rmse)
                    pcc_list.append(pcc)
                    rmse_indexlist.append(epoch)
                    pcc_indexlist.append(epoch)
                    print('TEST RMSE {:5.4f} PCC {:5.4f} '.format(rmse, pcc))
                """
                if args.mylog:
                    writer.add_scalars('data/loss', {'train': train_loss}, epoch )
                    writer.add_scalars('data/loss', {'val': val_loss}, epoch)
                    writer.add_scalars('data/mae', {'val': mae}, epoch)
                    writer.add_scalars('data/rmse', {'val': rmse_states}, epoch)
                    writer.add_scalars('data/rmse_states', {'val': rmse_states}, epoch)
                    writer.add_scalars('data/pcc', {'val': pcc}, epoch)
                    writer.add_scalars('data/pcc_states', {'val': pcc_states}, epoch)
                    writer.add_scalars('data/R2', {'val': r2}, epoch)
                    writer.add_scalars('data/R2_states', {'val': r2_states}, epoch)
                    writer.add_scalars('data/var', {'val': var}, epoch)
                    writer.add_scalars('data/var_states', {'val': var_states}, epoch)
                    writer.add_scalars('data/peak_mae', {'val': peak_mae}, epoch)
                """
                # Save the model if the validation loss is the best we've seen so far.
                """
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    bad_counter = 0
                    model_path = '%s/%s.pt' % (args.save_dir, log_token)
                    with open(model_path, 'wb') as f:
                        torch.save(model.state_dict(), f)
                    print('Best validation epoch:',epoch, time.ctime());
                    print("graph : ",np.round_(graph,2))
                    test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test,graph,tag='test')
                    rmse_list.append(rmse)
                    pcc_list.append(pcc)
                    rmse_indexlist.append(epoch)
                    pcc_indexlist.append(epoch)
                    print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
                else:
                    bad_counter += 1

                if bad_counter == args.patience:
                    break
                """
            print("---------------------------")
            print("best_rmse :", min(rmse_list))
            print("best_rmse_epoch :", rmse_indexlist[np.argmin(rmse_list)])
            print("---------------------------")
            print("best_pcc :", max(pcc_list))
            print("best_pcc_epoch :", pcc_indexlist[np.argmax(pcc_list)])
            print("---------------------------")

            f = open('./performance_record/{0}_{1}.csv'.format(args.dataset, args.horizon),'a',newline="", encoding='utf-8')
            wr = csv.writer(f)
            wr.writerow(["best_rmse: ", min(rmse_list)])
            wr.writerow(["best_rmse_epoch: ", rmse_indexlist[np.argmin(rmse_list)]])
            wr.writerow(["best_pcc: ", max(pcc_list)])
            wr.writerow(["best_pcc_epoch: ", pcc_indexlist[np.argmax(pcc_list)]])
            f.close()


        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early, epoch',epoch)

"""
# Load the best saved model.
model_path = '%s/%s.pt' % (args.save_dir, log_token)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f));
test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test,tag='test')
print('Final evaluation')
print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states,r2, r2_states, var, var_states, peak_mae))
"""
