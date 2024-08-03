import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import dgl


from feeder import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False):
        """

        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples

        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M



        # load data

        self.data = np.load(self.data_path)
        self.label = np.load(self.label_path)

        if self.debug:
            self.label = self.label[0:36]
            self.data = self.data[0:36]


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]




        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        source_node = []
        target_node = []

        for node in range(data_numpy.size(1) - 1):
            source_node.append(node)
            target_node.append(node + 1)

        source_node = torch.tensor(source_node)
        target_node = torch.tensor(target_node)

        edges = (source_node, target_node)
        G = dgl.graph(edges)
        G.ndata['feature'] = torch.tensor(each_graph_feature_array)

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)



if __name__ == '__main__':


    data_path = r"F:\system\Desktop\2s-AGCN-master - 副本\data\ntu\xsub\val_data_joint.npy"
    label_path = r"F:\system\Desktop\2s-AGCN-master - 副本\data\ntu\xsub\val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'


