import copy
from email.policy import default
from enum import Enum
import torch
import argparse
from torch_geometric import data
from torch_geometric.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss
def get_irm_loss(causal_pred, labels, batch_env_idx, criterion=F.cross_entropy):
    device = causal_pred.device
    dummy_w = torch.tensor(1.).to(device).requires_grad_()
    loss_0 = criterion(causal_pred[batch_env_idx == 0] * dummy_w, labels[batch_env_idx == 0])
    loss_1 = criterion(causal_pred[batch_env_idx == 1] * dummy_w, labels[batch_env_idx == 1])
    grad_0 = torch.autograd.grad(loss_0, dummy_w, create_graph=True)[0]
    grad_1 = torch.autograd.grad(loss_1, dummy_w, create_graph=True)[0]
    irm_loss = torch.sum(grad_0 * grad_1)

    return irm_loss


def get_contrast_loss(causal_rep, labels, norm=None, contrast_t=1.0, sampling='mul', y_pred=None, env_label=False):

    if norm != None:
        causal_rep = F.normalize(causal_rep)
    if sampling.lower() in ['mul', 'var']:
        # modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        # mask = mask.repeat(anchor_count, contrast_count)
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        is_valid = mask.sum(1) != 0
        mean_log_prob_pos = (mask * log_prob).sum(1)[is_valid] / mask.sum(1)[is_valid]
        # some classes may not be sampled by more than 2
        # mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0.0
        if is_valid.sum()==0:
            print("No contrastive samples found")
            return torch.tensor(0).to(labels.device)
        # loss
        contrast_loss = -mean_log_prob_pos.mean()
    elif sampling.lower() == 'gala':
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        # mask = mask.repeat(anchor_count, contrast_count)
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # find hard postive & negative
        pos_mask = torch.not_equal(y_pred.unsqueeze(1), y_pred.unsqueeze(1).T).float().to(device)
        neg_mask = torch.eq(y_pred.unsqueeze(1), y_pred.unsqueeze(1).T).float().to(device)

        
        # get rid of same pred label && same ground truth label
        logits_mask = logits_mask * torch.logical_not(mask*neg_mask)
        # get rid of diff pred label && diff ground truth
        # thus only hard negative & hard positive remains
        logits_mask = logits_mask * torch.logical_not(pos_mask*torch.logical_not(mask))
        # hard positive: same pred label && diff ground truth label
        pos_mask = mask * pos_mask
        assert (pos_mask*logits_mask).sum() == pos_mask.sum()
        if (pos_mask.sum(1)==0).sum() or (logits_mask.sum(1)==0).sum():
            print(f"#invalid pos {(pos_mask.sum(1)==0).sum()}, #invalid  neg {(logits_mask.sum(1)==0).sum()}")
        if pos_mask.sum()==0:
            print("Can't find positive samples")
            pos_mask = mask
        if logits_mask.sum() == 0:
            print("Can't find negative samples")
            logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+1e-12)

        # compute mean of log-likelihood over positive
        is_valid = pos_mask.sum(1) != 0
        mean_log_prob_pos = (pos_mask * log_prob).sum(1)[is_valid] / pos_mask.sum(1)[is_valid]
        contrast_loss = -mean_log_prob_pos.mean()
    else:
        raise Exception("Not implmented contrasting method")
    return contrast_loss
