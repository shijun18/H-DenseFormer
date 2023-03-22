import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np



class CrossentropyLoss(torch.nn.CrossEntropyLoss):

    def forward(self, inp, target):
        if target.size()[1] > 1:
            target = torch.argmax(target,1)
        target = target.long()
        num_classes = inp.size()[1]

        if len(inp.size()) == 5:
            inp = inp.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
        else:
            inp = inp.permute(0, 2, 3,  1).contiguous().view(-1, num_classes)
        target = target.view(-1,)

        return super(CrossentropyLoss, self).forward(inp, target)



class TopKLoss(CrossentropyLoss):

    def __init__(self, weight=None, ignore_index=-100, k=10, reduction=None):
        self.k = k
        self.reduction = reduction
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        loss, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        
        if self.reduction == "mean":
            loss = res.mean()
        elif self.reduction == "sum":
            loss = res.sum()

        return loss

class FocalLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs,dim=1)
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        p_t = (inputs * targets) + ((1 - inputs) * (1 - targets))
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss




class FLLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum"):
        super(FLLoss, self).__init__()
        self.eps = 1e-5
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs.clamp(self.eps, 1 - self.eps)

        ce_loss = - targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)

        p_t = (inputs * targets) + ((1 - inputs) * (1 - targets))
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss