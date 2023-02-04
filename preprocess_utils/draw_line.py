import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_name = 'PeMSD4'

    if data_name == 'PeMSD4':
        file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PeMSD4"))
        file_path = os.path.join(file_dir, 'PeMSD4.npz')
        kg_path = os.path.join(file_dir, 'PeMSD4.csv')
        num_of_vertices = 307
        node_idx = list(range(num_of_vertices))
    elif data_name == 'PeMSD8':
        file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PeMSD8"))
        file_path = os.path.join(file_dir, 'PeMSD8.npz')
        kg_path = os.path.join(file_dir, 'PeMSD8.csv')
        num_of_vertices = 170
        node_idx = list(range(num_of_vertices))
    elif data_name == 'PEMS03':
        file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PEMS03"))
        file_path = os.path.join(file_dir, 'pemsd3.npz')
        kg_path = os.path.join(file_dir, 'pemsd3.csv')
        # with open(os.path.join(file_dir, 'PEMS03.txt'), "r") as f:
        #     node_idx = f.readlines()
        # node_idx = [int(s) for s in node_idx]
        num_of_vertices = 358
        node_idx = list(range(num_of_vertices))
    elif data_name == 'PEMS07':
        file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/PEMS07"))
        file_path = os.path.join(file_dir, 'PEMS07.npz')
        kg_path = os.path.join(file_dir, 'PEMS07.csv')
        num_of_vertices = 883
        node_idx = list(range(num_of_vertices))

    data = np.load(file_path)['data'][:, :, 0]
    # node_list = [180, 183, 324]
    node_list = [107, 129, 130, 134]
    plt.figure(figsize=(10, 4.8))
    for i in node_list:
        y = data[0:32, i]
        color_list = ['r', 'g', 'b']
        x = np.arange(len(y))

        plt.plot(x, y, label='station: ' + str(i))
    plt.legend(loc=1, fontsize=10, frameon=True)
    plt.show()

