import torch, torch.nn as nn
import torch.nn.functional as functional


class DiceLoss(nn.Module):
    """
    Dice Loss
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, labels):

        # Compute dice loss
        prediction = torch.sigmoid(logits)
        dice_num = 2.0 * (prediction * labels).sum(dim=(2, 3))
        dice_den = (prediction * prediction + labels * labels).sum(dim=(2, 3)) + self.eps
        dice = 1.0 - (dice_num / dice_den).mean()

        return dice


class FocalLossLogits(nn.Module):
    """
    Binary focal loss on BCE logits (for class imbalance)
    :param alpha:
    : param gamma:
    """

    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma  # gamma = 0 -> BCE
        self.reduction = reduction

    def forward(self, logits, labels):
        # Compute binary cross entropy (after sigmoid)
        BCE = functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')

        # pt = prediction if labels=1 else (1-prediction)
        prediction = torch.sigmoid(logits)
        pt = prediction * labels + (1 - prediction) * (1 - labels)

        # alpha weighting per class
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        focal = alpha_t * (1 - pt).pow(self.gamma) * BCE

        # todo change reduction methods? leave as none?
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

class FocalDice(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLossLogits(alpha=alpha, gamma=gamma)
        self.dice_w = dice_weight
        self.focal_w = focal_weight

    def forward(self, logits, y):
        return self.dice_w * self.dice(logits, y) + self.focal_w * self.focal(logits, y)



