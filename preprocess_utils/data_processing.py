import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tqdm import tqdm
import random
import os
import torch
import json
import sys
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import eigs

sys.path.append(os.path.join(sys.path[0], '..'))


def split_multi_input(your_array):
    length = your_array.shape[0]
    time_series = []
    embedding = []
    for i in range(length):
        time_series.append(your_array[i][0])
        embedding.append(your_array[i][1])
    embedding = np.array(embedding)
    time_series = np.array(time_series)
    return time_series, embedding


def split_three_modal_input(your_array):
    length = your_array.shape[0]
    time_series = []
    embedding = []
    event_emb = []
    for i in range(length):
        time_series.append(your_array[i][0])
        embedding.append(your_array[i][1])
        event_emb.append(your_array[i][2])
    embedding = np.array(embedding)
    time_series = np.array(time_series)
    event_emb = np.array(event_emb)
    return time_series, embedding, event_emb


def split_kge_ts_input(your_array, with_distance=False):
    if with_distance:
        return your_array[:, 0], your_array[:, 1], your_array[:, 2]
    else:
        return your_array[:, 0], your_array[:, 1]


def add_multi_column(ts):
    temp_seq = []

    temp = np.array(ts).reshape(-1, 1)
    # normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    m_temp = scaler.fit_transform(temp)
    temp_seq.append(m_temp)

    # change rate
    temp_ts = []
    for i in range(len(ts) - 1):

        if ts[-(i + 2)] != 0:
            temp_ts.insert(0, (ts[-(i + 1)] - ts[-(i + 2)]) / ts[-(i + 2)])
        else:
            temp_ts.insert(0, (ts[-(i + 1)] - ts[-(i + 2)]) / 1)

    temp_ts.insert(0, 0)
    temp_seq.append(np.array(temp_ts).reshape(-1, 1))

    return temp_seq


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def scaled_Laplacian_old(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K=2):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def ts_process(ts, i, time_window, ts_multi_column):
    temp_seq = []
    c_temp = np.array(ts[i:i + time_window])

    c_temp = np.array(c_temp).reshape(-1, 1)

    temp_seq.append(c_temp)

    for multi_ts in ts_multi_column:
        temp_seq.append(multi_ts[i:i + time_window])

    temp_seq = np.array(temp_seq)
    temp_seq = np.squeeze(temp_seq)
    temp_seq = np.transpose(temp_seq)
    return temp_seq


class Data_loader:
    def __init__(self):
        """
        valid_list: list of raw_stock_data, if stock list provided by users contains invalid stock code, use valid list
                    to exclude those invalid code
        raw_stock_data: list of pd dataframe which is from data loader
        stock_list: index file

        """
        # initialization
        self.raw_stock_data = []
        self.valid_list = []
        self.group_ts = []
        self.distance = []

        # self.valid_list = None
        # self.raw_stock_data = None
        # self.group_ts = None
        # index_location = os.path.abspath(os.path.join(os.getcwd(), "../kg_enhanced multivariate ts/", "data"))
        # index_file = os.path.join(index_location, 'stock_index.json')
        # self.stock_list = json.load(open(index_file))

    def read_data(self, start_index=None, end_index=None, index_list=None, num_node=2, data_name='PeMSD4',
                  with_graph=False, hop=1):

        if data_name == 'PeMSD4':
            self.file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PeMSD4"))
            file_path = os.path.join(self.file_dir, 'PeMSD4.npz')
            kg_path = os.path.join(self.file_dir, 'PeMSD4.csv')
            self.num_of_vertices = 307
            node_idx = list(range(self.num_of_vertices))
        elif data_name == 'PeMSD8':
            self.file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PeMSD8"))
            file_path = os.path.join(self.file_dir, 'PeMSD8.npz')
            kg_path = os.path.join(self.file_dir, 'PeMSD8.csv')
            self.num_of_vertices = 170
            node_idx = list(range(self.num_of_vertices))
        elif data_name == 'PEMS03':
            self.file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PEMS03"))
            file_path = os.path.join(self.file_dir, 'pemsd3.npz')
            kg_path = os.path.join(self.file_dir, 'pemsd3.csv')
            # with open(os.path.join(self.file_dir, 'PEMS03.txt'), "r") as f:
            #     node_idx = f.readlines()
            # node_idx = [int(s) for s in node_idx]
            self.num_of_vertices = 358
            node_idx = list(range(self.num_of_vertices))
        elif data_name == 'PEMS07':
            self.file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PEMS07"))
            file_path = os.path.join(self.file_dir, 'PEMS07.npz')
            kg_path = os.path.join(self.file_dir, 'PEMS07.csv')
            self.num_of_vertices = 883
            node_idx = list(range(self.num_of_vertices))

        self.with_graph = with_graph

        if self.with_graph:
            data = np.load(file_path)['data'][:, :, 0]
            self.raw_stock_data = data[:, :].T
            adj_mx, distance_mx = get_adjacency_matrix(distance_df_filename=kg_path,
                                                       num_of_vertices=self.num_of_vertices)
            L_tilde = scaled_Laplacian(adj_mx)
            self.cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor) for i in
                                     cheb_polynomial(L_tilde)]
            self.adj_mx = torch.from_numpy(adj_mx).type(torch.FloatTensor)
            self.L_tilde = torch.from_numpy(L_tilde).type(torch.FloatTensor)

        else:
            data = np.load(file_path)['data'][:, :, 0]
            all_relation = pd.read_csv(kg_path)

            if start_index is not None and end_index is not None:
                if start_index < end_index:
                    temp_list = range(start_index, end_index)
                    node_idx_dic = [node_idx[idx] for idx in temp_list]
                else:
                    temp_list = range(self.num_of_vertices)
                    node_idx_dic = [node_idx[idx] for idx in temp_list]
                    # empty temp_list
            elif index_list is not None:
                temp_list = index_list
                node_idx_dic = [node_idx[idx] for idx in temp_list]

            for i in tqdm(temp_list):
                temp = data[:, i].T
                if temp is not None:
                    self.raw_stock_data.append(temp)

                in_node = all_relation[all_relation['to'] == node_idx[i]]['from']
                out_node = all_relation[all_relation['from'] == node_idx[i]]['to']
                nodes = list(set(list(in_node) + list(out_node)))
                temp=[]
                for _ in range(1, hop):
                  for n in nodes:
                      in_node = all_relation[all_relation['to'] == node_idx[n]]['from']
                      out_node = all_relation[all_relation['from'] == node_idx[n]]['to']
                      temp += list(set(list(in_node) + list(out_node)))
                nodes = list(set(temp+nodes).difference(set([node_idx[i]])))

                # print(nodes)
                # 存在相邻节点数量过少的情况
                # print("ts:" + str(i) + " " + "gts: " + str(nodes))
                if len(nodes) < num_node:
                    nodes += [node_idx[i]] * (num_node - len(nodes))
                else:
                    nodes = random.sample(nodes, num_node)

                distance = []
                # for n in nodes:
                #     if n in list(in_node):
                #         temp = all_relation[all_relation['from'] == n]
                #         temp = list(temp[temp['to'] == node_idx[i]]['distance'])[0]
                #         distance.append(1 / temp + 1)
                #     elif n == node_idx[i]:
                #         distance.append(0)
                #     else:
                #         temp = all_relation[all_relation['to'] == n]
                #         temp = list(temp[temp['from'] == node_idx[i]]['distance'])[0]
                #         distance.append(1 / temp)
                self.distance.append(distance)

                node_ts = []
                # todo 对节点进行权重排序？
                # code in this
                # -------

                for j in nodes:
                    temp = data[:, node_idx.index(j)].T
                    if temp is not None:
                        if len(temp) == 0:
                            node_ts.append(self.raw_stock_data[-1])
                        else:
                            node_ts.append(temp)
                    else:
                        print('Not found ', j, ' , please check data.')
                        return
                self.group_ts.append(node_ts)
            self.distance = np.array(self.distance)

        return

    def data_processing_kge_ts_input(self, time_window=14):
        if os.path.exists(self.file_dir + '/' + str(time_window) + '_prepocessed_data.npz') and self.with_graph:
            datas = np.load(self.file_dir + '/' + str(time_window) + '_prepocessed_data.npz', allow_pickle=True)
            x_set = datas['x']
            y_set = datas['y']
            return x_set, y_set

        # group_ts = np.expand_dims(self.raw_stock_data, 0).repeat(self.num_of_vertices, axis=0)
        x_set = []
        y_set = []
        if self.with_graph:
            for index in tqdm(range(len(self.raw_stock_data))):
                ts = self.raw_stock_data[index]
                for i in range(len(ts) - time_window):
                    y_set.append(ts[i + time_window])
                # raw_data_group = group_ts[index]
                # ts = [x for x in raw_data]
                # add multi-column
                ts_multi_column = [ts.reshape(-1, 1)]
                ts_multi_column += add_multi_column(ts)
                ts_multi_column = np.array(ts_multi_column)
                ts_multi_column = np.squeeze(ts_multi_column)
                ts_multi_column = np.transpose(ts_multi_column)
                x_set.append(ts_multi_column)
            x_set = np.array(x_set)
            x_set = np.expand_dims(x_set, 0).repeat(self.num_of_vertices, axis=0)
            x_set = x_set.transpose(0, 2, 1, 3)  # num_node, seq_len, num_node, 3
            temp_ts = []
            for i in tqdm(range(self.num_of_vertices)):
                for j in range(x_set.shape[1] - time_window):
                    temp_ts.append(np.squeeze(x_set[i, j:j + time_window, :, :]))
                    # y_set.append(x_set[i + time_window, :, 0])
            temp_ts = np.array(temp_ts)
            x_set = np.array([temp_ts, []])
            y_set = np.array(y_set)
            y_set = y_set.reshape(-1, 1)

            # ts_group = []
            # ts_group_multi_column = []


        else:
            for index in tqdm(range(len(self.raw_stock_data))):
                raw_data = self.raw_stock_data[index]
                raw_data_group = self.group_ts[index]
                ts = [x for x in raw_data]
                # add multi-column

                ts_multi_column = add_multi_column(ts)
                ts_group = []
                ts_group_multi_column = []
                for j in range(len(raw_data_group)):
                    temp = raw_data_group[j]
                    temp = [x for x in temp]
                    ts_group.append(temp)
                    ts_group_multi_column.append(add_multi_column(temp))

                for i in range(len(ts) - time_window):
                    temp_seq = []
                    temp = np.array(ts[i:i + time_window])

                    temp = np.array(temp).reshape(-1, 1)
                    temp_seq.append(temp)

                    for multi_ts in ts_multi_column:
                        temp_seq.append(multi_ts[i:i + time_window])

                    temp_seq = np.array(temp_seq)
                    temp_seq = np.squeeze(temp_seq)
                    temp_seq = np.transpose(temp_seq)

                    temp_group = []
                    for j in range(len(ts_group)):
                        temp_group.append(ts_process(ts_group[j], i, time_window, ts_multi_column))

                    x_set.append(np.array([temp_seq, np.array(temp_group), self.distance[index]]))
                    y_set.append(ts[i + time_window])
            x_set = np.array(x_set)
            y_set = np.array(y_set)

        return x_set, y_set


if __name__ == '__main__':
    graph = pd.read_csv("../data/PEMS03/PEMS03.csv")
    '''
    total_node = set(graph['from']).union(set(graph['to']))
    node_dic = {}
    for n in total_node:
        node_dic[n] = len(graph[graph['from'] == n]['to']) + len(graph[graph['to'] == n]['from'])
    print(np.mean(list(node_dic.values())))'''
    a = graph[graph['from'] == 317842][['to', 'distance']]
    print(list(a)[:][0])

    # test = Stock_loader()
    # test.choose_stock(start_index=0, end_index=10)
    # embeddings_loc = '../data/kg_embeddings/mp2v_32d_0510.csv'
    # embeddings = test.get_embeddings(data_loc=embeddings_loc)
    # build dataset for multi_input model
    """
    x_set, y_set = test.data_processing_multi_input(embeddings_list=embeddings)
    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.2, random_state=0)
    x_train_t, x_train_e = split_multi_input(x_train)
    x_test_t, x_test_e = split_multi_input(x_test)
    """
    # build dataset for only price model
    """
    x_set, y_set = test.data_processing()
    """
    # build dataset for para concat model
    """
    x_set, y_set = test.data_processing_para(embeddings_list=embeddings)
    """
    # build dataset for price seq model
    # x_set, y_set = test.data_processing_seq()