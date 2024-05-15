"""
    Most borrow from: https://github.com/Alibaba-MIIL/ASL
"""
import torch
import torch.nn as nn


def bce_loss(x, y):
    y = torch.clamp(y, min=0, max=1)
    return nn.BCEWithLogitsLoss()(x, y)

def masked_bce_loss(x, y):
    mask = y != 0
    y = torch.clamp(y, min=0, max=1)
    bce = nn.BCEWithLogitsLoss(reduction='none')(x, y)
    bce = bce * mask
    return bce.sum() / mask.sum()

def ranking_loss(logitsT, labelsT):

    # Refer: https://github.com/akshitac8/BiAM

    eps = 1e-8
    subset_idxT = torch.sum(torch.abs(labelsT), dim=0)
    subset_idxT = torch.nonzero(subset_idxT > 0).view(-1).long().cuda()
    sub_labelsT = labelsT[:, subset_idxT]
    sub_logitsT = logitsT[:, subset_idxT]
    positive_tagsT = torch.clamp(sub_labelsT, 0., 1.)
    negative_tagsT = torch.clamp(-sub_labelsT, 0., 1.)
    maskT = positive_tagsT.unsqueeze(1) * negative_tagsT.unsqueeze(-1)
    pos_score_matT = sub_logitsT * positive_tagsT
    neg_score_matT = sub_logitsT * negative_tagsT
    IW_pos3T = pos_score_matT.unsqueeze(1)
    IW_neg3T = neg_score_matT.unsqueeze(-1)
    OT = 1 + IW_neg3T - IW_pos3T
    O_maskT = maskT * OT
    diffT = torch.clamp(O_maskT, 0)
    violationT = torch.sign(diffT).sum(1).sum(1)
    diffT = diffT.sum(1).sum(1)
    lossT = torch.mean(diffT / (violationT + eps))

    return lossT


def matching_loss(scores, labels):
    return nn.BCELoss()(scores, labels)


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        y = torch.clamp(y, min=0, max=1) # to binary label [-1, 1] --> [0, 1]
        targets = y
        anti_targets = 1 - y

        # Calculating Probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss.add_(anti_targets * torch.log(xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    xs_pos = xs_pos * targets
                    xs_neg = xs_neg * anti_targets
                    asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                                self.gamma_pos * targets + self.gamma_neg * anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                loss *= asymmetric_w
            else:
                xs_pos = xs_pos * targets
                xs_neg = xs_neg * anti_targets
                asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                            self.gamma_pos * targets + self.gamma_neg * anti_targets)   
                loss *= asymmetric_w         
        _loss = - loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss
    
    
class MaskedASL(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(MaskedASL, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        mask = y != 0
        y = torch.clamp(y, min=0, max=1)  # to binary label [-1, 1] --> [0, 1]
        
        targets = y
        anti_targets = 1 - y

        # Calculating Probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss.add_(anti_targets * torch.log(xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    xs_pos = xs_pos * targets
                    xs_neg = xs_neg * anti_targets
                    asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                                self.gamma_pos * targets + self.gamma_neg * anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                loss *= asymmetric_w
            else:
                xs_pos = xs_pos * targets
                xs_neg = xs_neg * anti_targets
                asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                            self.gamma_pos * targets + self.gamma_neg * anti_targets)   
                loss *= asymmetric_w
                
        # print((-loss * mask).sum(), mask.sum())
        # print((-loss).sum(), loss.shape[0] * loss.shape[1])   
        
        loss = -loss * mask
        loss = loss.sum() / mask.sum() 
        
        return loss