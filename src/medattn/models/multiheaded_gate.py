import torch.nn as nn, torch.nn.functional as F

class MultiHeadAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_heads=4):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, 1, bias=True)
        self.bn_g = nn.BatchNorm2d(F_int); self.bn_x = nn.BatchNorm2d(F_int)
        self.psi = nn.Conv2d(F_int, num_heads, 1, bias=True)
        self.combine = nn.Conv2d(num_heads, 1, 1, bias=True)

    def forward(self, x, g):
        g1 = self.bn_g(self.W_g(g))
        x1 = self.bn_x(self.W_x(x))
        g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        a  = (g1 + x1).relu()
        heads = self.psi(a).sigmoid()
        attn  = self.combine(heads).sigmoid()
        return x * attn, attn, heads
