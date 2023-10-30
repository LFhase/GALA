import torch.nn as nn
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing

def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            #PyG 2.0.4
            module._explain = True
            module._edge_mask = mask
            module._apply_sigmoid = False
            #PyG 1.7.2
            module.__explain__ = True
            module.__edge_mask__ = mask

# def set_masks(mask: Tensor, model: nn.Module):
#     r"""
#     Modified from https://github.com/wuyxin/dir-gnn.
#     """
#     for mmodule in model.modules():
#         if isinstance(mmodule, MessagePassing):
#             module = mmodule
#     module._apply_sigmoid = False
#     module.__explain__ = True
#     module._explain = True
#     module.__edge_mask__ = mask
#     module._edge_mask = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            #PyG 2.0.4
            module._explain = False
            module._edge_mask = None
            # module._apply_sigmoid = True
            #PyG 1.7.2
            module.__explain__ = False
            module.__edge_mask__ = None
