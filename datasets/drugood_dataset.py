import os.path as osp
import pickle as pkl

import torch
import random
from tqdm import tqdm
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, add_self_loops
import copy

class DrugOOD(InMemoryDataset):

    def __init__(self, root, dataset, name, mode, transform=None, pre_transform=None, pre_filter=None):
        """
        init from original DrugOOD dataset in the form of dgl.heterograph.DGLHeteroGraph
        """
        self.root = root
        self.dataset = dataset
        self.name = name
        self.mode = mode
        super(DrugOOD, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_data(root, dataset, name, mode)

    def load_data(self, root, dataset, name, mode, get_data_list=False):
        data_path = osp.join(root, name + "_" + mode + ".pt")
        if not osp.exists(data_path) or get_data_list:
            data_list = []
            # for data in dataset:
            for step, data in tqdm(enumerate(dataset), total=len(dataset), desc="Converting"):
                graph = data['input']
                y = data['gt_label']
                group = data['group']

                edge_index = graph.edges()
                edge_attr = graph.edata['x']  #.long()
                node_attr = graph.ndata['x']  #.long()
                new_data = Data(edge_index=torch.stack(list(edge_index), dim=0),
                                edge_attr=edge_attr,
                                x=node_attr,
                                y=y,
                                group=group)
                data_list.append(new_data)
            if get_data_list:
                return data_list
            torch.save(self.collate(data_list), data_path)

        self.data, self.slices = torch.load(data_path)
        
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
        super(DrugOOD, self).__init__(self.root)
        self.data, self.slices = self.collate(data_list)
        # self.orig_data = copy.deepcopy(self._data)
        # self._data_list = data_list
        # self._data = self.data
        # print(len(self._data_list))
        print(f"new #sum{len(data_list)} #pos{num_pos} #neg{num_neg+len(neg_data_list)*(num_repeats-1)}")
    def resume_samples(self):
        data_list = self.load_data(self.root, self.dataset, self.name, self.mode,get_data_list=True)
        super(DrugOOD, self).__init__(self.root)
        self.data, self.slices = self.collate(data_list)
