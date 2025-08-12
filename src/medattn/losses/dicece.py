import torch, torch.nn as nn

class DiceCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.BCEWithLogitsLoss()
    def forward(self, logits, y):
        ce = self.ce(logits, y)
        p = torch.sigmoid(logits)
        eps = 1e-6
        num = 2*(p*y).sum(dim=(2,3))
        den = (p*p + y*y).sum(dim=(2,3)) + eps
        dice = 1 - (num/den).mean()
        return ce + dice