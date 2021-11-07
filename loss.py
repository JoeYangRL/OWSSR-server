import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules import loss

"""
# used args paramters
args.T [check]
"""

def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.range(0, logits_open.size(0) - 1).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo

def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    return Le

def dist_loss(args, logits_m, logits_open_m, logits_old, logits_open_old):

    # 1. closed head
    loss1 = nn.KLDivLoss()(F.log_softmax(logits_m/args.T), F.softmax(logits_old/args.T)) * args.T * args.T
    # 2. open head
    logits_open_m = logits_open_m.view(logits_open_m.size(0), 2, -1)
    logits_open_old = logits_open_old.view(logits_open_old.size(0), 2, -1)
    loss2 = nn.KLDivLoss()(F.log_softmax(logits_open_m/args.T), F.softmax(logits_open_old/args.T)) * args.T * args.T

    return loss1+loss2
    

