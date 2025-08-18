import torch, torch.nn as nn

class DiceCE(nn.Module):
    """
    Compound loss of Dice + CE for our binary image segmentation
    """
    def __init__(self):
        super().__init__()
        self.LogitsCE = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):

        # Compute binary cross entropy (after sigmoid)
        BCE = self.LogitsCE(logits, labels)

        # Compute dice loss
        prediction = torch.sigmoid(logits)
        eps = 1e-6
        dice_num = 2.0 * (prediction * labels).sum(dim=(2, 3))
        dice_den = (prediction * prediction + labels * labels).sum(dim=(2, 3)) + eps
        dice = 1.0 - (dice_num / dice_den).mean()

        compound_loss = BCE + dice
        return compound_loss
