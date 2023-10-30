import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.batch as DataBatch
from torch_geometric.nn import (ASAPooling, global_add_pool, global_max_pool,
                                global_mean_pool)
from utils.get_subgraph import relabel, split_batch
from utils.mask import clear_masks, set_masks

from models.conv import GNN_node, GNN_node_Virtualnode
from models.gnn import GNN, LeGNN


class GNNERM(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean"):
        super(GNNERM, self).__init__()
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=emb_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              edge_dim=edge_dim)

    def forward(self, batch, return_data="pred"):
        causal_pred, causal_rep = self.classifier(batch, get_rep=True)
        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            #Nothing will happen for ERM
            return causal_pred, causal_rep
        else:
            raise Exception("Not support return type")

import torch_scatter
from torch.distributions.normal import Normal
def bce_log(pred, gt, eps=1e-8):
    prob = torch.sigmoid(pred)
    return -(gt * torch.log(prob + eps) + (1 - gt) * torch.log(1 - prob + eps))

def discrete_gaussian(nums, std=1):
    Dist = Normal(loc=0, scale=1)
    plen, halflen = std * 6 / nums, std * 3 / nums
    posx = torch.arange(-3 * std + halflen, 3 * std, plen)
    result = Dist.cdf(posx + halflen) - Dist.cdf(posx - halflen)
    return result / result.sum()
def KLDist(p, q, eps=1e-8):
    log_p, log_q = torch.log(p + eps), torch.log(q + eps)
    return torch.sum(p * (log_p - log_q))

class GNNEnv(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 num_envs=2,
                 prior="uniform"):
        super(GNNEnv, self).__init__()
        self.gnn = GNN(gnn_type=gnn_type,
                              input_dim=input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=emb_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              edge_dim=edge_dim)
        self.num_envs = num_envs
        self.num_tasks = out_dim
        # env inference
        self.env_pred_linear = torch.nn.Linear(emb_dim+1, num_envs)
        # conditional gnn
        self.class_emb = torch.nn.Parameter(
            torch.zeros(num_envs, emb_dim)
        )
        self.env_label_pred_linear = torch.nn.Linear(emb_dim + emb_dim, out_dim)
        # main gnn
        self.graph_label_pred_linear = torch.nn.Linear(emb_dim, out_dim)
        if prior == 'uniform':
            self.prior = torch.ones(self.num_envs) / self.num_envs
        else:
            self.prior = discrete_gaussian(self.num_envs)

    def get_env_loss(self,batch,criterion):
        h_graph = self.gnn.forward_rep(batch)
        y_part = torch.nan_to_num(batch.y).float().unsqueeze(1)
        env_prob = self.env_pred_linear(torch.cat([h_graph, y_part], dim=-1))
        q_e = torch.softmax(env_prob, dim=-1)
        batch_size = h_graph.size(0)
        device = h_graph.device
        losses = []
        for dom in range(self.num_envs):
            domain_info = torch.ones(batch_size).long().to(device)
            domain_feat = torch.index_select(self.class_emb, 0, domain_info*dom)
            p_ye = self.env_label_pred_linear(torch.cat([h_graph, domain_feat], dim=1))
            labeled = batch.y == batch.y
            # there are nan in the labels so use this to mask them
            # and this is a multitask binary classification
            # data_belong = torch.arange(batch_size).long()
            # data_belong = data_belong.unsqueeze(dim=-1).to(device)
            # data_belong = data_belong.repeat(1, self.num_tasks)
            # [batch_size, num_tasks] same as p_ye
            loss = criterion(p_ye[labeled], batch.y[labeled],reduction='none')
            # shape: [numbers of not nan gts]
            # batch_loss = torch_scatter.scatter(
            #     loss, dim=0, index=data_belong[labeled],
            #     reduce='mean'
            # )  # [batch_size]
            # considering the dataset is a multitask binary
            # classification task, the process above is to
            # get a average loss among all the tasks,
            # when there is only one task, it's equilvant to
            # bce_with_logit without reduction
            losses.append(loss)
        losses = torch.stack(losses, dim=1)  # [batch_size, num_domain]
        Eq = torch.mean(torch.sum(q_e * losses, dim=-1))
        ELBO = Eq + KLDist(q_e, self.prior.to(device))
        return ELBO
    def forward_env(self,batch,criterion):
        batch_size = batch.y.size(0)
        device =  batch.y.device
        labeled = batch.y == batch.y
        data_belong = torch.arange(batch_size).long()
        data_belong = data_belong.unsqueeze(dim=-1).to(device)
        data_belong = data_belong.repeat(1, self.num_tasks)
        with torch.no_grad():
            self.eval()
            h_graph = self.gnn.forward_rep(batch)
            cond_result = []
            for dom in range(self.num_envs):
                domain_info = torch.ones(batch_size).long().to(device)
                # domain_info = (domain_info * dom).to(device)
                domain_feat = torch.index_select(self.class_emb, 0, domain_info*dom)
                cond_term = criterion(
                    self.env_label_pred_linear(torch.cat([h_graph, domain_feat], dim=1))[labeled],
                    batch.y[labeled],
                    reduction='none'
                )
                # cond_term = torch_scatter.scatter(
                #     cond_term, dim=0, index=data_belong[labeled],
                #     reduce='mean'
                # )
                cond_result.append(cond_term)
            cond_result = torch.stack(cond_result, dim=0)
            # [num_domain, batch_size]
            cond_result = torch.matmul(self.prior.to(device), cond_result)
            # cond_result = torch.mean(cond_result, dim=0)
            # [batch_size]

            y_part = torch.nan_to_num(batch.y).unsqueeze(1).float()
            env_prob = self.env_pred_linear(torch.cat([h_graph, y_part], dim=-1))
            env = torch.argmax(env_prob, dim=-1)
            # [batch_size]
        return env, cond_result, data_belong

    def forward(self, batch, return_data="pred"):
        causal_pred, causal_rep = self.gnn(batch, get_rep=True)
        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            #Nothing will happen for ERM
            return causal_pred, causal_rep
        else:
            raise Exception("Not support return type")


class GNNPooling(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 pooling='asap',
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean"):
        super(GNNPooling, self).__init__()
        if pooling.lower() == 'asap':
            # Cancel out the edge attribute when using ASAP pooling
            # since (1) ASAP not compatible with edge attr
            #       (2) performance of DrugOOD will not be affected w/o edge attr
            self.pool = ASAPooling(emb_dim, ratio, dropout=drop_ratio)
            edge_dim = -1
        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
        self.ratio = ratio
        self.pooling = pooling

        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=emb_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=emb_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              edge_dim=edge_dim)

    def forward(self, batched_data, return_data="pred"):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        device = x.device
        h = self.gnn_encoder(batched_data)
        edge_weight = None  #torch.ones(edge_index[0].size()).to(device)
        x, edge_index, causal_edge_weight, batch, perm = self.pool(h, edge_index, edge_weight=edge_weight, batch=batch)
        col, row = batched_data.edge_index
        node_mask = torch.zeros(batched_data.x.size(0)).to(device)
        node_mask[perm] = 1
        edge_mask = node_mask[col] * node_mask[row]
        if self.pooling.lower() == 'asap':
            # Cancel out the edge attribute when using ASAP pooling
            # since (1) ASAP not compatible with edge attr
            #       (2) performance of DrugOOD will not be affected w/o edge attr
            edge_attr = torch.ones(row.size()).to(device)

        # causal_x, causal_edge_index, causal_batch, _ = relabel(x, edge_index, batch)
        causal_x, causal_edge_index, causal_batch = x, edge_index, batch
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)
        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            #Nothing will happen for ERM
            return causal_pred, causal_rep
        else:
            raise Exception("Not support return type")

from torch_geometric.nn import InstanceNorm
class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

class CIGA(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 c_dim=-1,
                 c_in="raw",
                 c_rep="rep",
                 c_pool="add",
                 s_rep="rep",
                 pretrain=20):
        super(CIGA, self).__init__()
        ## GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
        self.ratio = ratio
        # self.edge_att = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, 1))
        self.edge_att = MLP([emb_dim * 2, emb_dim * 4, emb_dim, 1], dropout=drop_ratio)
        self.s_rep = s_rep
        self.c_rep = c_rep
        self.c_pool = c_pool

        # predictor based on the decoupled subgraph
        self.pred_head = "spu" if s_rep.lower() == "conv" else "inv"
        self.c_dim = emb_dim if c_dim < 0 else c_dim
        self.c_in = c_in
        self.c_input_dim = input_dim if c_in.lower() == "raw" else emb_dim
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=self.c_input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=self.c_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              pred_head=self.pred_head,
                              edge_dim=edge_dim)
        # if c_in=='raw':
        #     self.gnn_encoder = self.classifier.gnn_node
        self.cal_fw = None
        self.log_sigmas = nn.Parameter(torch.zeros(3))
        self.log_sigmas.requires_grad_(True)
        print(self)

    def split_graph(self,data, edge_score, ratio):
        from torch_geometric.utils import degree
        def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
            r'''
            Adopt from <https://github.com/rusty1s/pytorch_scatter/issues/48>_.
            '''
            f_src = src.float()
            f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
            norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
            perm = norm.argsort(dim=dim, descending=descending)

            return src[perm], perm

        def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
            rank, perm = sparse_sort(src, index, dim, descending, eps)
            num_nodes = degree(index, dtype=torch.long)
            k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
            start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
            mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
            mask = torch.cat(mask, dim=0)
            mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
            topk_perm = perm[mask]
            exc_perm = perm[~mask]

            return topk_perm, exc_perm, rank, perm, mask

        has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None
        new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
        new_causal_edge_index = data.edge_index[:, new_idx_reserve]
        new_spu_edge_index = data.edge_index[:, new_idx_drop]

        new_causal_edge_weight = edge_score[new_idx_reserve]
        new_spu_edge_weight = edge_score[new_idx_drop]

        if has_edge_attr:
            new_causal_edge_attr = data.edge_attr[new_idx_reserve]
            new_spu_edge_attr = data.edge_attr[new_idx_drop]
        else:
            new_causal_edge_attr = None
            new_spu_edge_attr = None

        return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
            (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight)
    
    
    def forward(self, batch, pred_edge_weight=None, return_data="pred", return_spu=False, debug=False):
        device = batch.x.device
        # obtain the graph embeddings from the featurizer GNN encoder
        h = self.gnn_encoder(batch)
        # seperate the input graphs into \hat{G_c} and \hat{G_s}
        # using edge-level attetion
        row, col = batch.edge_index
        if batch.edge_attr == None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        if pred_edge_weight is None:
            pred_edge_weight = self.edge_att(edge_rep,batch.batch[col]).view(-1).sigmoid()
        if self.ratio<0:
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = (batch.edge_index, batch.edge_attr, pred_edge_weight), \
                (batch.edge_index, batch.edge_attr, pred_edge_weight)
        else:
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = self.split_graph(batch, pred_edge_weight, self.ratio)


        if self.c_in.lower() == "raw":
            causal_x, causal_edge_index, causal_batch, _ = relabel(batch.x, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(batch.x, spu_edge_index, batch.batch)
        else:
            causal_x, causal_edge_index, causal_batch, _ = relabel(h, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(h, spu_edge_index, batch.batch)

        # obtain \hat{G_c}
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        # obtain predictions with the classifier based on \hat{G_c}
        causal_pred, causal_rep = self.classifier(causal_graph,get_rep=True)
        clear_masks(self.classifier)
        
        # whether to return the \hat{G_s} for further use
        if return_spu:
            spu_graph = DataBatch.Batch(batch=spu_batch,
                                         edge_index=spu_edge_index,
                                         x=spu_x,
                                         edge_attr=spu_edge_attr)
            set_masks(1-spu_edge_weight, self.classifier)
            if self.s_rep.lower() == "conv":
                spu_pred, spu_rep = self.classifier.get_spu_pred_forward(spu_graph, get_rep=True,grad=True)
            else:
                spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True,grad=True)
            clear_masks(self.classifier)
            causal_pred = (causal_pred, spu_pred)

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "attention":
            return pred_edge_weight
        elif return_data.lower() == "feat":
            causal_h, _, __, ___ = relabel(h, causal_edge_index, batch.batch)
            if self.c_pool.lower() == "add":
                casual_rep_from_feat = global_add_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "max":
                casual_rep_from_feat = global_max_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "mean":
                casual_rep_from_feat = global_mean_pool(causal_h, batch=causal_batch)
            else:
                raise Exception("Not implemented contrastive feature pooling")

            return causal_pred, casual_rep_from_feat
        else:
            return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight

    def get_dir_loss(self, batched_data, labels, criterion, is_labeled=None, return_data="pred"):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight = self.forward(batched_data,return_data="")
        if is_labeled == None:
            is_labeled = torch.ones(labels.size()).to(labels.device)

        def get_comb_pred(predictor, causal_graph_x, spu_graph_x):
            causal_pred = predictor.graph_pred_linear(causal_graph_x)
            spu_pred = predictor.spu_mlp(spu_graph_x).detach()
            return torch.sigmoid(spu_pred) * causal_pred

        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, edge_attr=spu_edge_attr)
        set_masks(1-spu_edge_weight, self.classifier)
        spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
        clear_masks(self.classifier)

        env_loss = torch.tensor([]).to(causal_rep.device)
        for spu in spu_rep:
            rep_out = get_comb_pred(self.classifier, causal_rep, spu)
            env_loss = torch.cat([env_loss, criterion(rep_out[is_labeled], labels[is_labeled]).unsqueeze(0)])

        dir_loss = torch.var(env_loss * spu_rep.size(0)) + env_loss.mean()

        if return_data.lower() == "pred":
            return get_comb_pred(causal_rep, spu_rep)
        elif return_data.lower() == "rep":
            return dir_loss, causal_pred, spu_pred, causal_rep
        else:
            return dir_loss, causal_pred
    
    def get_grea_loss(self, batched_data, labels, criterion, is_labeled=None, return_data="pred"):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight = self.forward(batched_data,return_data="")
        if is_labeled == None:
            is_labeled = torch.ones(labels.size()).to(labels.device)

        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, edge_attr=spu_edge_attr)
        set_masks(1-spu_edge_weight, self.classifier)
        spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
        clear_masks(self.classifier)

        # compile grea augmentation
        hidden_dim = causal_rep.size(-1)
        # (batch_size)^2 x hidden_dim
        grea_rep = (causal_rep.unsqueeze(1)+spu_rep.unsqueeze(0)).view(-1,hidden_dim)
        grea_loss = self.classifier.forward_spu_cls(grea_rep)
        target_rep = batched_data.y.repeat_interleave(batched_data.batch[-1]+1,dim=0)
        is_labeled_rep = target_rep == target_rep
        grea_loss = criterion(grea_rep[is_labeled_rep],target_rep[is_labeled_rep])

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return grea_loss, causal_pred, spu_pred, causal_rep
        else:
            return grea_loss, causal_pred
    def get_cal_loss(self, batched_data, labels, criterion, is_labeled=None, return_data="pred"):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight = self.forward(batched_data,return_data="")
        if self.cal_fw is None:
            emb_dim = self.classifier.emb_dim
            num_class = self.classifier.num_class
            self.cal_fw = torch.nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                                                         nn.Linear(2 * emb_dim, num_class)).to(causal_x.device)
        if is_labeled == None:
            is_labeled = torch.ones(labels.size()).to(labels.device)

        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, edge_attr=spu_edge_attr)
        set_masks(1-spu_edge_weight, self.classifier)
        spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True,grad=True)
        clear_masks(self.classifier)

        # compile grea augmentation
        hidden_dim = causal_rep.size(-1)
        # (batch_size)^2 x hidden_dim
        grea_rep = (causal_rep.unsqueeze(1)+spu_rep.unsqueeze(0)).view(-1,hidden_dim)
        grea_loss = self.cal_fw(grea_rep)
        target_rep = batched_data.y.repeat_interleave(batched_data.batch[-1]+1,dim=0)
        is_labeled_rep = target_rep == target_rep
        grea_loss = criterion(grea_rep[is_labeled_rep],target_rep[is_labeled_rep])

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return grea_loss, causal_pred, spu_pred, causal_rep
        else:
            return grea_loss, causal_pred
    def get_disc_loss(self, batched_data, labels, criterion, is_labeled=None, return_data="pred",grad=False):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight = self.forward(batched_data,return_data="")
        if is_labeled == None:
            is_labeled = torch.ones(labels.size()).to(labels.device)

        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, edge_attr=spu_edge_attr)
        set_masks(1-spu_edge_weight, self.classifier)
        spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True,grad=grad)
        clear_masks(self.classifier)
        
        # we would compile disc loss in the main loop
        disc_loss = 0
        # compile grea augmentation
        # hidden_dim = causal_rep.size(-1)
        # (batch_size)^2 x hidden_dim
        # grea_rep = (causal_rep.unsqueeze(1)+spu_rep.unsqueeze(0)).view(-1,hidden_dim)
        # disc_loss = self.classifier.forward_spu_cls(grea_rep)
        # target_rep = batched_data.y.repeat_interleave(batched_data.batch[-1]+1,dim=0)
        # is_labeled_rep = target_rep == target_rep
        # disc_loss = criterion(grea_rep[is_labeled_rep],target_rep[is_labeled_rep])

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return disc_loss, causal_pred, spu_pred, causal_rep
        elif return_data.lower() == "spu":
            return disc_loss, causal_pred, spu_pred, causal_rep, spu_rep
        else:
            return disc_loss, causal_pred


class GALA(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 c_dim=-1,
                 c_in="raw",
                 c_rep="rep",
                 c_pool="add",
                 s_rep="rep",
                 pretrain=20):
        super(GALA, self).__init__()
        ### GNN to generate node embeddings
        # if gnn_type.lower() == "le":
        #     self.gnn_encoder = LeGNN(in_channels=input_dim,
        #                              hid_channels=emb_dim,
        #                              num_layer=num_layers,
        #                              drop_ratio=drop_ratio,
        #                              num_classes=out_dim,
        #                              edge_dim=edge_dim)
        # else:
        #     if virtual_node:
        #         self.gnn_encoder = GNN_node_Virtualnode(num_layers,
        #                                                 emb_dim,
        #                                                 input_dim=input_dim,
        #                                                 JK=JK,
        #                                                 drop_ratio=drop_ratio,
        #                                                 residual=residual,
        #                                                 gnn_type=gnn_type,
        #                                                 edge_dim=edge_dim)
        #     else:
        #         self.gnn_encoder = GNN_node(num_layers,
        #                                     emb_dim,
        #                                     input_dim=input_dim,
        #                                     JK=JK,
        #                                     drop_ratio=drop_ratio,
        #                                     residual=residual,
        #                                     gnn_type=gnn_type,
        #                                     edge_dim=edge_dim)
        self.ratio = ratio
        self.edge_att = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, 1),nn.Sigmoid())
        self.s_rep = s_rep
        self.c_rep = c_rep
        self.c_pool = c_pool

        # predictor based on the decoupled subgraph
        self.pred_head = "spu" if s_rep.lower() == "conv" else "inv"
        self.c_dim = emb_dim if c_dim < 0 else c_dim
        self.c_in = c_in
        self.c_input_dim = input_dim if c_in.lower() == "raw" else emb_dim
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=self.c_input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=self.c_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              pred_head=self.pred_head,
                              edge_dim=edge_dim)
        self.gnn_encoder = self.classifier.gnn_node
        self.log_sigmas = nn.Parameter(torch.zeros(3))
        self.log_sigmas.requires_grad_(True)

    def split_graph(self,data, edge_score, ratio):
        from torch_geometric.utils import degree
        def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
            r'''
            Adopt from <https://github.com/rusty1s/pytorch_scatter/issues/48>_.
            '''
            f_src = src.float()
            f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
            norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
            perm = norm.argsort(dim=dim, descending=descending)

            return src[perm], perm

        def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
            rank, perm = sparse_sort(src, index, dim, descending, eps)
            num_nodes = degree(index, dtype=torch.long)
            k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
            start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
            mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
            mask = torch.cat(mask, dim=0)
            mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
            topk_perm = perm[mask]
            exc_perm = perm[~mask]

            return topk_perm, exc_perm, rank, perm, mask

        has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None
        new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
        new_causal_edge_index = data.edge_index[:, new_idx_reserve]
        new_spu_edge_index = data.edge_index[:, new_idx_drop]

        new_causal_edge_weight = edge_score[new_idx_reserve]
        new_spu_edge_weight = edge_score[new_idx_drop]

        if has_edge_attr:
            new_causal_edge_attr = data.edge_attr[new_idx_reserve]
            new_spu_edge_attr = data.edge_attr[new_idx_drop]
        else:
            new_causal_edge_attr = None
            new_spu_edge_attr = None

        return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
            (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight)

    def forward(self, batch, return_data="pred", return_spu=False, debug=False):
        # obtain the graph embeddings from the featurizer GNN encoder
        with torch.no_grad():
            h = self.gnn_encoder(batch)
        device = h.device
        # seperate the input graphs into \hat{G_c} and \hat{G_s}
        # using edge-level attetion
        row, col = batch.edge_index
        if batch.edge_attr == None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        pred_edge_weight = self.edge_att(edge_rep).view(-1)
        if self.ratio<0:
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = (batch.edge_index, batch.edge_attr, pred_edge_weight), \
                (batch.edge_index, batch.edge_attr, pred_edge_weight)
        else:
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = self.split_graph(batch, pred_edge_weight, self.ratio)


        if self.c_in.lower() == "raw":
            causal_x, causal_edge_index, causal_batch, _ = relabel(batch.x, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(batch.x, spu_edge_index, batch.batch)
        else:
            causal_x, causal_edge_index, causal_batch, _ = relabel(h, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(h, spu_edge_index, batch.batch)

        # obtain \hat{G_c}
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        # obtain predictions with the classifier based on \hat{G_c}
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        # whether to return the \hat{G_s} for further use
        if return_spu:
            spu_graph = DataBatch.Batch(batch=spu_batch,
                                         edge_index=spu_edge_index,
                                         x=spu_x,
                                         edge_attr=spu_edge_attr)
            set_masks(1-spu_edge_weight, self.classifier)
            if self.s_rep.lower() == "conv":
                spu_pred, spu_rep = self.classifier.get_spu_pred_forward(spu_graph, get_rep=True)
            else:
                spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
            clear_masks(self.classifier)
            causal_pred = (causal_pred, spu_pred)

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            causal_h, _, __, ___ = relabel(h, causal_edge_index, batch.batch)
            if self.c_pool.lower() == "add":
                casual_rep_from_feat = global_add_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "max":
                casual_rep_from_feat = global_max_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "mean":
                casual_rep_from_feat = global_mean_pool(causal_h, batch=causal_batch)
            else:
                raise Exception("Not implemented contrastive feature pooling")

            return causal_pred, casual_rep_from_feat
        else:
            return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight

    def get_dir_loss(self, batched_data, labels, criterion, is_labeled=None, return_data="pred"):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight = self.forward(batched_data,return_data="")
        if is_labeled == None:
            is_labeled = torch.ones(labels.size()).to(labels.device)

        def get_comb_pred(predictor, causal_graph_x, spu_graph_x):
            causal_pred = predictor.graph_pred_linear(causal_graph_x)
            spu_pred = predictor.spu_mlp(spu_graph_x).detach()
            return torch.sigmoid(spu_pred) * causal_pred

        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, edge_attr=spu_edge_attr)
        set_masks(1-spu_edge_weight, self.classifier)
        spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
        clear_masks(self.classifier)

        env_loss = torch.tensor([]).to(causal_rep.device)
        for spu in spu_rep:
            rep_out = get_comb_pred(self.classifier, causal_rep, spu)
            env_loss = torch.cat([env_loss, criterion(rep_out[is_labeled], labels[is_labeled]).unsqueeze(0)])

        dir_loss = torch.var(env_loss * spu_rep.size(0)) + env_loss.mean()

        if return_data.lower() == "pred":
            return get_comb_pred(causal_rep, spu_rep)
        elif return_data.lower() == "rep":
            return dir_loss, causal_pred, spu_pred, causal_rep
        else:
            return dir_loss, causal_pred
    
    def get_grea_loss(self, batched_data, labels, criterion, is_labeled=None, return_data="pred"):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight = self.forward(batched_data,return_data="")
        if is_labeled == None:
            is_labeled = torch.ones(labels.size()).to(labels.device)

        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, edge_attr=spu_edge_attr)
        set_masks(1-spu_edge_weight, self.classifier)
        spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
        clear_masks(self.classifier)

        # compile grea augmentation
        hidden_dim = causal_rep.size(-1)
        # (batch_size)^2 x hidden_dim
        grea_rep = (causal_rep.unsqueeze(1)+spu_rep.unsqueeze(0)).view(-1,hidden_dim)
        grea_loss = self.classifier.forward_spu_cls(grea_rep)
        target_rep = batched_data.y.repeat_interleave(batched_data.batch[-1]+1,dim=0)
        is_labeled_rep = target_rep == target_rep
        grea_loss = criterion(grea_rep[is_labeled_rep],target_rep[is_labeled_rep])

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return grea_loss, causal_pred, spu_pred, causal_rep
        else:
            return grea_loss, causal_pred
    
    def get_disc_loss(self, batched_data, labels, criterion, is_labeled=None, return_data="pred"):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight = self.forward(batched_data,return_data="")
        if is_labeled == None:
            is_labeled = torch.ones(labels.size()).to(labels.device)

        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, edge_attr=spu_edge_attr)
        set_masks(1-spu_edge_weight, self.classifier)
        spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
        clear_masks(self.classifier)
        
        # we would compile disc loss in the main loop
        disc_loss = 0
        # compile grea augmentation
        # hidden_dim = causal_rep.size(-1)
        # (batch_size)^2 x hidden_dim
        # grea_rep = (causal_rep.unsqueeze(1)+spu_rep.unsqueeze(0)).view(-1,hidden_dim)
        # disc_loss = self.classifier.forward_spu_cls(grea_rep)
        # target_rep = batched_data.y.repeat_interleave(batched_data.batch[-1]+1,dim=0)
        # is_labeled_rep = target_rep == target_rep
        # disc_loss = criterion(grea_rep[is_labeled_rep],target_rep[is_labeled_rep])

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return disc_loss, causal_pred, spu_pred, causal_rep
        elif return_data.lower() == "spu":
            return disc_loss, causal_pred, spu_pred, causal_rep, spu_rep
        else:
            return disc_loss, causal_pred

class GSAT(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 c_dim=-1,
                 c_in="raw",
                 c_rep="rep",
                 c_pool="add",
                 s_rep="rep",
                 pretrain=20):
        super(GSAT, self).__init__()
        ### GNN to generate node embeddings
        # if gnn_type.lower() == "le":
        #     self.gnn_encoder = LeGNN(in_channels=input_dim,
        #                              hid_channels=emb_dim,
        #                              num_layer=num_layers,
        #                              drop_ratio=drop_ratio,
        #                              num_classes=out_dim,
        #                              edge_dim=edge_dim)
        # else:
        #     if virtual_node:
        #         self.gnn_encoder = GNN_node_Virtualnode(num_layers,
        #                                                 emb_dim,
        #                                                 input_dim=input_dim,
        #                                                 JK=JK,
        #                                                 drop_ratio=drop_ratio,
        #                                                 residual=residual,
        #                                                 gnn_type=gnn_type,
        #                                                 edge_dim=edge_dim)
        #     else:
        #         self.gnn_encoder = GNN_node(num_layers,
        #                                     emb_dim,
        #                                     input_dim=input_dim,
        #                                     JK=JK,
        #                                     drop_ratio=drop_ratio,
        #                                     residual=residual,
        #                                     gnn_type=gnn_type,
        #                                     edge_dim=edge_dim)
        self.ratio = ratio
        self.edge_att = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, 1))
        self.s_rep = s_rep
        self.c_rep = c_rep
        self.c_pool = c_pool

        # predictor based on the decoupled subgraph
        self.pred_head = "spu" if s_rep.lower() == "conv" else "inv"
        self.c_dim = emb_dim if c_dim < 0 else c_dim
        self.c_in = c_in
        self.c_input_dim = input_dim if c_in.lower() == "raw" else emb_dim
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=self.c_input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=self.c_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              pred_head=self.pred_head,
                              edge_dim=edge_dim)
        self.gnn_encoder = self.classifier.gnn_node

        self.init_r = 0.9
        self.decay_r = 0.1
        all_decay_r = self.init_r-self.ratio
        self.decay_interval = pretrain//2//(all_decay_r//self.decay_r)
        self.final_r = ratio

    def split_graph(self,data, edge_score, ratio):
        from torch_geometric.utils import degree
        def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
            r'''
            Adopt from <https://github.com/rusty1s/pytorch_scatter/issues/48>_.
            '''
            f_src = src.float()
            f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
            norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
            perm = norm.argsort(dim=dim, descending=descending)

            return src[perm], perm

        def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
            rank, perm = sparse_sort(src, index, dim, descending, eps)
            num_nodes = degree(index, dtype=torch.long)
            k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
            start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
            mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
            mask = torch.cat(mask, dim=0)
            mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
            topk_perm = perm[mask]
            exc_perm = perm[~mask]

            return topk_perm, exc_perm, rank, perm, mask

        has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None
        new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
        new_causal_edge_index = data.edge_index[:, new_idx_reserve]
        new_spu_edge_index = data.edge_index[:, new_idx_drop]

        new_causal_edge_weight = edge_score[new_idx_reserve]
        new_spu_edge_weight = edge_score[new_idx_drop]

        if has_edge_attr:
            new_causal_edge_attr = data.edge_attr[new_idx_reserve]
            new_spu_edge_attr = data.edge_attr[new_idx_drop]
        else:
            new_causal_edge_attr = None
            new_spu_edge_attr = None

        return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
            (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight)

    def forward(self, batch, return_data="pred", return_spu=False, debug=False):
        # obtain the graph embeddings from the featurizer GNN encoder
        h = self.gnn_encoder(batch)
        device = h.device
        # seperate the input graphs into \hat{G_c} and \hat{G_s}
        # using edge-level attetion
        row, col = batch.edge_index
        if batch.edge_attr == None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        pred_edge_weight = self.edge_att(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = (batch.edge_index, batch.edge_attr, pred_edge_weight), \
                (batch.edge_index, batch.edge_attr, pred_edge_weight)

        if self.c_in.lower() == "raw":
            causal_x, causal_edge_index, causal_batch, _ = relabel(batch.x, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(batch.x, spu_edge_index, batch.batch)
        else:
            causal_x, causal_edge_index, causal_batch, _ = relabel(h, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(h, spu_edge_index, batch.batch)

        # obtain \hat{G_c}
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        causal_edge_weight = self.concrete_sample(causal_edge_weight, temp=1, training=self.training)
        set_masks(causal_edge_weight, self.classifier)
        # obtain predictions with the classifier based on \hat{G_c}
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        # whether to return the \hat{G_s} for further use
        if return_spu:
            print("??")
            spu_graph = DataBatch.Batch(batch=spu_batch,
                                         edge_index=spu_edge_index,
                                         x=spu_x,
                                         edge_attr=spu_edge_attr)
            set_masks(1-spu_edge_weight, self.classifier)
            if self.s_rep.lower() == "conv":
                spu_pred, spu_rep = self.classifier.get_spu_pred_forward(spu_graph, get_rep=True)
            else:
                spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
            clear_masks(self.classifier)
            causal_pred = (causal_pred, spu_pred)

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            causal_h, _, __, ___ = relabel(h, causal_edge_index, batch.batch)
            if self.c_pool.lower() == "add":
                casual_rep_from_feat = global_add_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "max":
                casual_rep_from_feat = global_max_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "mean":
                casual_rep_from_feat = global_mean_pool(causal_h, batch=causal_batch)
            else:
                raise Exception("Not implemented contrastive feature pooling")

            return causal_pred, casual_rep_from_feat
        else:
            return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight
    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
            r = init_r - current_epoch // decay_interval * decay_r
            if r < final_r:
                r = final_r
            return r
    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
    def get_gsat_loss(self, epoch, batch, return_data="pred"):
        # (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
        #         (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
        #         pred_edge_weight = self.forward(batched_data,return_data="")
        h = self.gnn_encoder(batch)
        device = h.device
        # seperate the input graphs into \hat{G_c} and \hat{G_s}
        # using edge-level attetion
        row, col = batch.edge_index
        if batch.edge_attr == None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        pred_edge_weight = self.edge_att(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = (batch.edge_index, batch.edge_attr, pred_edge_weight), \
                (batch.edge_index, batch.edge_attr, pred_edge_weight)

        if self.c_in.lower() == "raw":
            causal_x, causal_edge_index, causal_batch, _ = relabel(batch.x, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(batch.x, spu_edge_index, batch.batch)
        else:
            causal_x, causal_edge_index, causal_batch, _ = relabel(h, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(h, spu_edge_index, batch.batch)

        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        att = self.concrete_sample(causal_edge_weight, temp=1, training=self.training)
        set_masks(att, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, edge_attr=spu_edge_attr)
        # set_masks(1-spu_edge_weight, self.classifier)
        # spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True, grad=True)
        # clear_masks(self.classifier)
        
        r = self.get_r(self.decay_interval, self.decay_r, epoch, init_r=self.init_r, final_r=self.final_r)
        gsat_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return gsat_loss, causal_pred, causal_pred, causal_rep
        else:
            return gsat_loss, causal_pred
