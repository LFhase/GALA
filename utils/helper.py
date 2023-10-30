
import torch
import numpy as np
import os
import random
from texttable import Texttable
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import subgraph
def plot_and_save_subgraph(args,model,test_loader, eval_model, eval_metric,evaluator,device):
    # if args.visualize:
    # if args.epoch == -1:
    #     model_path = os.path.join('pred', args.dataset) + f"{args.commit}.pt"
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    #     print("Loaded model from ", model_path)
    model.eval()
    test_acc, pred = eval_model(model, device, test_loader, evaluator, eval_metric=eval_metric, save_pred=True)
    print(f"model performance: {test_acc}")
    # print(pred[pred != 0])
    if eval_metric == 'rocauc':
        pred = torch.Tensor(pred > 0.5).long().view(-1)
    else:
        pred = torch.Tensor(pred).long().view(-1)
    if "spmotif" in args.dataset.lower():
        with torch.no_grad():
            full_pred = []
            for batch in test_loader:
                batch.to(device)
                cur_pred = model(batch)
                full_pred.append(F.softmax(cur_pred, dim=1).cpu())
        full_pred = torch.max(torch.cat(full_pred, dim=0), dim=1)[0].numpy()
    test_dataset = test_loader.dataset
    # idx = get_viz_idx(test_dataset, args.dataset, pred=pred, full_pred=full_pred)
    # idx = np.concatenate(idx)
    idx = np.random.randint(len(test_dataset), size=(120))
    viz_set = test_dataset[idx]
    data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
    # data = process_data(data, use_edge_attr)
    # batch_att, _, clf_logits = eval_one_batch(data.to(device), epoch)
    with torch.no_grad():
        batch_att = model(data.to(device), return_data="viz")[-1]
        batch_att = batch_att.cpu()
    # batch_att = causal_edge_weightq
    imgs = []
    for i in tqdm(range(len(viz_set))):
        mol_type, coor = None, None
        if args.dataset.lower() == 'mutag':
            node_dict = {
                0: 'C',
                1: 'O',
                2: 'Cl',
                3: 'H',
                4: 'N',
                5: 'F',
                6: 'Br',
                7: 'S',
                8: 'P',
                9: 'I',
                10: 'Na',
                11: 'K',
                12: 'Li',
                13: 'Ca'
            }
            mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
        elif args.dataset.lower() in ["graph-sst2", "graph-sst5", "graph-twitter", "graph-tt"]:
            mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
            num_nodes = data.x.shape[0]
            x = np.linspace(0, 1, num_nodes)
            y = np.ones_like(x)
            coor = np.stack([x, y], axis=1)
        elif args.dataset == 'ogbg_molhiv':
            element_idxs = {k: int(v + 1) for k, v in enumerate(viz_set[i].x[:, 0])}
            mol_type = {
                k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                for k, v in element_idxs.items()
            }
        elif 'drugood' in args.dataset.lower():
            symbol_list = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', '*']
            mol_type = {k: symbol_list[torch.where(v == 1)[0]] for k, v in enumerate(viz_set[i].x[:, :15])}
        elif args.dataset == 'mnist':
            raise NotImplementedError
        node_subset = data.batch == i
        _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att)
        # print(viz_set[i])
        if "spmotif" in args.dataset.lower():
            node_label = viz_set[i].node_label.reshape(-1)
        else:
            node_label = torch.zeros(viz_set[i].x.shape[0])
        fig, img = visualize_a_graph(viz_set[i].edge_index,
                                     edge_att,
                                     node_label,
                                     args.dataset,
                                     label=viz_set[i].y,
                                     norm=True,
                                     mol_type=mol_type,
                                     coor=coor,gsat=args.ginv_opt.lower()=="gsat",
                                     visualize_top=args.visualize_top)
        imgs.append(img)
        # plt.savefig(
        #     os.path.join("plots2", args.dataset) + args.commit + f"_{idx[i]}_label{viz_set[i].y.item()}.pdf")
        img_path = os.path.join("plots", args.dataset) + args.commit +f"{args.ginv_opt}_c{args.contrast}" + f"_label{viz_set[i].y.item()}_{idx[i]}"
        if args.visualize_top>0:
            img_path+=f"_vt{args.visualize_top}"
        plt.savefig(img_path+".png")
        plt.close()
    # imgs = np.stack(imgs)
    # self.writer.add_images(tag, imgs, epoch, dataformats='NHWC')

def get_viz_idx(test_set, dataset_name, pred=None, full_pred=None, num_viz_samples=10):
    if pred is not None:
        print(pred)
        print(test_set.data.y)
        if full_pred is not None:
            correct_idx = torch.nonzero((test_set.data.y == pred) * (full_pred > 0.95), as_tuple=True)[0]
        else:
            correct_idx = torch.nonzero(test_set.data.y == pred, as_tuple=True)[0]
        print(len(test_set))
        y_dist = test_set.data.y[correct_idx].numpy().reshape(-1)
        # print(len(test_set))
    else:
        y_dist = test_set.data.y.numpy().reshape(-1)
    print(len(y_dist))
    num_nodes = np.array([each.x.shape[0] for each in test_set])
    classes = np.unique(y_dist)
    res = []
    for each_class in classes:
        tag = 'class_' + str(each_class)
        if dataset_name.lower() in ["graph-sst2", "graph-sst5", "graph-twitter", "graph-tt"]:
            condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
            candidate_set = np.nonzero(condi)[0]
        else:
            candidate_set = np.nonzero(y_dist == each_class)[0]
        idx = np.random.choice(candidate_set, num_viz_samples, replace=False)
        # res.append((idx, tag))
        res.append(correct_idx[idx])
    return res


import matplotlib.pyplot as plt
import matplotlib
from rdkit import Chem
import torch
import random
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
# use type-1 font
plt.switch_backend('agg')
# plt.rcParams['pdf.use14corefonts'] = True
# font = {'size': 16, 'family': 'Helvetica'}
# plt.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16


def visualize_a_graph(edge_index,
                      edge_att,
                      node_label,
                      dataset_name,
                      label,
                      coor=None,
                      norm=False,
                      mol_type=None,
                      nodesize=300,gsat=False,visualize_top=-1):
    plt.clf()
    plt.title(f"{dataset_name.replace('ec','ic')}: y={int(label)}")
    if norm:
        if gsat==True:
            edge_att = edge_att**10
            edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)
        else:
            edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)
            # print(edge_att)
            edge_att = edge_att**2.5
            edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)
            # edge_att = F.softmax(edge_att)
        print(sum(edge_att), sum(edge_att > 0.5))
    if visualize_top>0:
        _, indices = torch.sort(edge_att,descending=True)
        edge_att_new = edge_att.clone()
        edge_att_new[indices[:visualize_top]] = 0
        edge_att = edge_att-edge_att_new
    elif visualize_top==0:
        edge_att = torch.ones(edge_att.size())/2

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00', 'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
    # G = to_networkx(data, edge_attrs=['att'])

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '',
            xy=pos[target],
            xycoords='data',
            xytext=pos[source],
            textcoords='data',
            arrowprops=dict(
                arrowstyle="->" if dataset_name == 'Graph-SST2' else '-',
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.4' if dataset_name == 'Graph-SST2' else 'arc3'))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G,
                               pos,
                               width=1,
                               edge_color='gray',
                               arrows=False,
                               alpha=0.1,
                               ax=ax,
                               connectionstyle='arc3,rad=0.4')

    fig = plt.gcf()
    fig.canvas.draw()
    plt.tight_layout()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return fig, image


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_partition(len_dataset, device, seed, p=[0.5, 0.5]):
    '''
        group the graph randomly

        Input:   len_dataset   -> [int]
                 the number of data to be groupped
                 
                 device        -> [torch.device]
                
                 p             -> [list]
                 probabilities of the random assignment for each group
        Output: 
                 vec           -> [torch.LongTensor]
                 group assignment for each data
    '''
    assert abs(np.sum(p) - 1) < 1e-4

    vec = torch.tensor([]).to(device)
    for idx, idx_p in enumerate(p):
        vec = torch.cat([vec, torch.ones(int(len_dataset * idx_p)).to(device) * idx])

    vec = torch.cat([vec, torch.ones(len_dataset - len(vec)).to(device) * idx])
    perm = torch.randperm(len_dataset, generator=torch.Generator().manual_seed(seed))
    return vec.long()[perm]


def args_print(args, logger):
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())


def PrintGraph(graph):

    if graph.name:
        print("Name: %s" % graph.name)
    print("# Nodes:%6d      | # Edges:%6d |  Class: %2d" \
          % (graph.num_nodes, graph.num_edges, graph.y))

    print("# Node features: %3d| # Edge feature(s): %3d" \
          % (graph.num_node_features, graph.num_edge_features))
