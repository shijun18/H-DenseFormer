import torch.nn as nn
import torch.nn.functional as F

from loss.dice_loss import DiceLoss
from loss.cross_entropy import CrossentropyLoss, FocalLoss

#---------------------------------seg loss---------------------------------
class CEPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus cross entropy
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(CEPlusDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict,target)

        ce = CrossentropyLoss(weight=self.weight)
        ce_loss = ce(predict,target)
        
        total_loss = ce_loss + dice_loss

        return total_loss

class FLPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus cross entropy
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(FLPlusDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict,target)

        ce = FocalLoss(reduction="mean")
        ce_loss = ce(predict,target)
        
        total_loss = ce_loss + dice_loss

        return total_loss



class DeepSuperloss(nn.Module):
    def __init__(self, criterion=None):
        super(DeepSuperloss, self).__init__()
        self.loss = criterion
    def forward(self, input, target):
        loss = 0
        for i, img in enumerate(input):
            w = 1 / (2 ** i)
            label = F.interpolate(target, img.size()[2:])
            l = self.loss(img, label)
            loss += l * w
        return loss 