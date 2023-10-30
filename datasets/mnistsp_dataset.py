# adapt from https://github.com/bknyaz/graph_attention_pool/blob/master/graphdata.py
import numpy as np
import os.path as osp
import pickle
import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data
import copy

def compute_adjacency_matrix_images(coord, sigma=0.1):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(-dist / (sigma * np.pi)**2)
    A[np.diag_indices_from(A)] = 0
    return A


def list_to_torch(data):
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data


class CMNIST75sp(InMemoryDataset):
    splits = ['test', 'train']

    def __init__(self,
                 root,
                 mode='train',
                 use_mean_px=True,
                 use_coord=True,
                 node_gt_att_threshold=0,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        self.node_gt_att_threshold = node_gt_att_threshold
        self.use_mean_px, self.use_coord = use_mean_px, use_coord
        super(CMNIST75sp, self).__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index('cmnist_75sp_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['cmnist_75sp_train.pkl', 'cmnist_75sp_test.pkl']

    @property
    def processed_file_names(self):
        return ['cmnist_75sp_train.pt', 'cmnist_75sp_test.pt']

    def download(self):
        for file in self.raw_file_names:
            if not osp.exists(osp.join(self.raw_dir, file)):
                print("raw data of `{}` doesn't exist, please download from our github.".format(file))
                raise FileNotFoundError

    def process(self,get_data_list=False):

        data_file = 'cmnist_75sp_%s.pkl' % self.mode
        with open(osp.join(self.raw_dir, data_file), 'rb') as f:
            self.labels, self.sp_data = pickle.load(f)

        self.use_mean_px = self.use_mean_px
        self.use_coord = self.use_coord
        self.n_samples = len(self.labels)
        self.img_size = 28
        self.node_gt_att_threshold = self.node_gt_att_threshold

        self.edge_indices, self.xs, self.edge_attrs, self.node_gt_atts, self.edge_gt_atts = [], [], [], [], []
        data_list = []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord = sample[:2]
            coord = coord / self.img_size
            A = compute_adjacency_matrix_images(coord)
            N_nodes = A.shape[0]

            A = torch.FloatTensor((A > 0.1) * A)
            edge_index, edge_attr = dense_to_sparse(A)

            x = None
            if self.use_mean_px:
                x = mean_px.reshape(N_nodes, -1)
            if self.use_coord:
                coord = coord.reshape(N_nodes, 2)
                if self.use_mean_px:
                    x = np.concatenate((x, coord), axis=1)
                else:
                    x = coord
            if x is None:
                x = np.ones(N_nodes, 1)  # dummy features

            # replicate features to make it possible to test on colored images
            x = np.pad(x, ((0, 0), (2, 0)), 'edge')
            if self.node_gt_att_threshold == 0:
                node_gt_att = (mean_px > 0).astype(np.float32)
            else:
                node_gt_att = mean_px.copy()
                node_gt_att[node_gt_att < self.node_gt_att_threshold] = 0

            node_gt_att = torch.LongTensor(node_gt_att).view(-1)
            row, col = edge_index
            edge_gt_att = torch.LongTensor(node_gt_att[row] * node_gt_att[col]).view(-1)

            data_list.append(
                Data(x=torch.tensor(x),
                     y=torch.LongTensor([self.labels[index]]),
                     edge_index=edge_index,
                     edge_attr=edge_attr,
                     node_gt_att=node_gt_att,
                     edge_gt_att=edge_gt_att,
                     name=f'CMNISTSP-{self.mode}-{index}',
                     idx=index))
        if get_data_list:
            return data_list
        idx = self.processed_file_names.index('cmnist_75sp_{}.pt'.format(self.mode))
        torch.save(self.collate(data_list), self.processed_paths[idx])
    def rebalance_samples(self,is_pos,repeats=-1):
        num_pos = is_pos.sum().item()
        num_neg = is_pos.size(0)-num_pos
        print(f"Rebalancing")
        print(f"original #pos{num_pos} #neg{num_neg}")
        is_pos *= num_pos>num_neg
        num_pos = is_pos.sum().item()
        num_neg = is_pos.size(0)-num_pos
        num_repeats = min(num_pos//num_neg,2)
        if repeats>0:
            num_repeats = repeats+1
        # data_list = self.process(get_data_list=True)
        data_list = copy.deepcopy(self._data_list)
        neg_position = torch.nonzero(torch.logical_not(is_pos),as_tuple=True)[0].tolist()
        # neg_position = torch.nonzero(is_pos,as_tuple=True)[0].tolist()
        assert len(neg_position)==num_neg
        neg_data_list = [data_list[idx] for idx in neg_position]
        # data_list = neg_data_list*5 #neg_data_list*(num_repeats-1)+data_list
        data_list = neg_data_list*(num_repeats-1)+data_list
        super(CMNIST75sp, self).__init__(self.root)
        self.data, self.slices = self.collate(data_list)
        # self.orig_data = copy.deepcopy(self._data)
        # self._data_list = data_list
        # self._data = self.data
        # print(len(self._data_list))
        print(f"new #sum{len(data_list)} #pos{num_pos} #neg{num_neg+len(neg_data_list)*(num_repeats-1)}")
    def resume_samples(self):
        data_list = self.process(get_data_list=True)
        super(CMNIST75sp, self).__init__(self.root)
        self.data, self.slices = self.collate(data_list)

class MNIST75sp(InMemoryDataset):
    splits = ['test', 'train']

    def __init__(self,
                 root,
                 mode='train',
                 use_mean_px=True,
                 use_coord=True,
                 node_gt_att_threshold=0,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        self.node_gt_att_threshold = node_gt_att_threshold
        self.use_mean_px, self.use_coord = use_mean_px, use_coord
        super(MNIST75sp, self).__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index('mnist_75sp_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['mnist_75sp_train.pkl', 'mnist_75sp_test.pkl']

    @property
    def processed_file_names(self):
        return ['mnist_75sp_train.pt', 'mnist_75sp_test.pt']

    def download(self):
        for file in self.raw_file_names:
            if not osp.exists(osp.join(self.raw_dir, file)):
                print("raw data of `{}` doesn't exist, please download from our github.".format(file))
                raise FileNotFoundError

    def process(self):

        data_file = 'mnist_75sp_%s.pkl' % self.mode
        with open(osp.join(self.raw_dir, data_file), 'rb') as f:
            self.labels, self.sp_data = pickle.load(f)

        self.use_mean_px = self.use_mean_px
        self.use_coord = self.use_coord
        self.n_samples = len(self.labels)
        self.img_size = 28
        self.node_gt_att_threshold = self.node_gt_att_threshold

        self.edge_indices, self.xs, self.edge_attrs, self.node_gt_atts, self.edge_gt_atts = [], [], [], [], []
        data_list = []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord = sample[:2]
            coord = coord / self.img_size
            A = compute_adjacency_matrix_images(coord)
            N_nodes = A.shape[0]

            A = torch.FloatTensor((A > 0.1) * A)
            edge_index, edge_attr = dense_to_sparse(A)

            x = None
            if self.use_mean_px:
                x = mean_px.reshape(N_nodes, -1)
            if self.use_coord:
                coord = coord.reshape(N_nodes, 2)
                if self.use_mean_px:
                    x = np.concatenate((x, coord), axis=1)
                else:
                    x = coord
            if x is None:
                x = np.ones(N_nodes, 1)  # dummy features

            # replicate features to make it possible to test on colored images
            x = np.pad(x, ((0, 0), (2, 0)), 'edge')
            if self.node_gt_att_threshold == 0:
                node_gt_att = (mean_px > 0).astype(np.float32)
            else:
                node_gt_att = mean_px.copy()
                node_gt_att[node_gt_att < self.node_gt_att_threshold] = 0

            node_gt_att = torch.LongTensor(node_gt_att).view(-1)
            row, col = edge_index
            edge_gt_att = torch.LongTensor(node_gt_att[row] * node_gt_att[col]).view(-1)

            data_list.append(
                Data(x=torch.tensor(x),
                     y=torch.LongTensor([self.labels[index]]),
                     edge_index=edge_index,
                     edge_attr=edge_attr,
                     node_gt_att=node_gt_att,
                     edge_gt_att=edge_gt_att,
                     name=f'MNISTSP-{self.mode}-{index}',
                     idx=index))
        idx = self.processed_file_names.index('mnist_75sp_{}.pt'.format(self.mode))
        torch.save(self.collate(data_list), self.processed_paths[idx])


# class MNIST75sp(torch.utils.data.Dataset):
#     def __init__(self,
#                  data_dir,
#                  split,
#                  use_mean_px=True,
#                  use_coord=True,
#                  node_gt_att_threshold=0):

#         self.data_dir = data_dir
#         self.split = split
#         self.is_test = split.lower() in ['test', 'val']
#         with open(osp.join(data_dir, 'mnist_75sp_%s.pkl' % split), 'rb') as f:
#             self.labels, self.sp_data = pickle.load(f)

#         self.use_mean_px = use_mean_px
#         self.use_coord = use_coord
#         self.n_samples = len(self.labels)
#         self.img_size = 28
#         self.node_gt_att_threshold = node_gt_att_threshold

#         print('loading the %s set...' % self.split.upper())
#         self.edge_indices, self.xs, self.edge_attrs, self.node_gt_atts, self.edge_gt_atts = [], [], [], [], []
#         self.graphs = []
#         for index, sample in enumerate(self.sp_data):
#             mean_px, coord = sample[:2]
#             coord = coord / self.img_size
#             A = compute_adjacency_matrix_images(coord)
#             N_nodes = A.shape[0]

#             A = torch.FloatTensor((A > 0.05) * A)
#             edge_index, edge_attr = dense_to_sparse(A)

#             x = None
#             if self.use_mean_px:
#                 x = mean_px.reshape(N_nodes, -1)
#             if self.use_coord:
#                 coord = coord.reshape(N_nodes, 2)
#                 if self.use_mean_px:
#                     x = np.concatenate((x, coord), axis=1)
#                 else:
#                     x = coord
#             if x is None:
#                 x = np.ones(N_nodes, 1)  # dummy features

#             # replicate features to make it possible to test on colored images
#             x = np.pad(x, ((0, 0), (2, 0)), 'edge')
#             if self.node_gt_att_threshold == 0:
#                 node_gt_att = (mean_px > 0).astype(np.float32)
#             else:
#                 node_gt_att = mean_px.copy()
#                 node_gt_att[node_gt_att < self.node_gt_att_threshold] = 0

#             node_gt_att = torch.LongTensor(node_gt_att).view(-1)
#             row, col = edge_index
#             edge_gt_att = torch.LongTensor(node_gt_att[row] * node_gt_att[col]).view(-1)

#             self.graphs.append(
#                 Data(
#                     x=torch.tensor(x),
#                     y=torch.LongTensor([self.labels[index]]),
#                     edge_index=edge_index,
#                     edge_attr=edge_attr,
#                     node_gt_att=node_gt_att,
#                     edge_gt_att=edge_gt_att,
#                     name=f'MNISTSP-{self.split}-{index}', idx=index
#                 )
#             )

#     def train_val_split(self, samples_idx):
#         self.sp_data = [self.sp_data[i] for i in samples_idx]
#         self.labels = self.labels[samples_idx]
#         self.n_samples = len(self.labels)

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, index):

#         return self.graphs[index]
