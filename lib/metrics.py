# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2023-6-21
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2023 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import numpy as np


def compute_AP(scores, labels):
    num_class = scores.size(1)
    ap = torch.zeros(num_class).to(scores.device)
    for idx_cls in range(num_class):
        prediction = scores[:, idx_cls]
        label = labels[:, idx_cls]
        mask = label.abs() == 1
        if (label > 0).sum() == 0:
            continue
        binary_label = torch.clamp(label[mask], min=0, max=1)
        sorted_pred, sort_idx = prediction[mask].sort(descending=True)
        sorted_label = binary_label[sort_idx]
        tmp = (sorted_label == 1).float()
        tp = tmp.cumsum(0)
        fp = (sorted_label != 1).float().cumsum(0)
        num_pos = binary_label.sum()
        rec = tp / num_pos
        prec = tp / (tp + fp)
        ap_cls = (tmp * prec).sum() / num_pos
        ap[idx_cls].copy_(ap_cls)
    mAP = torch.mean(ap).item()
    return mAP


def compute_F1(scores, labels, k_val):
    device = scores.device
    idx = scores.topk(dim=1, k=k_val)[1]
    scores = torch.zeros_like(scores, device=device)
    scores.scatter_(dim=1, index=idx, src=torch.ones(idx.shape, device=device))
    mask = scores == 1
    TP = (labels[mask] == 1).sum().float()
    tpfp = mask.sum().float()
    tpfn = (labels == 1).sum().float()
    p = TP / tpfp
    r = TP / tpfn
    f1 = 2 * p * r / (p + r)
    p = torch.mean(p).item()
    r = torch.mean(r).item()
    f1 = torch.mean(f1).item()
    
    return p, r, f1
