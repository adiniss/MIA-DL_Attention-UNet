import torch.nn as nn
import torch.nn.functional as functional


class MultiHeadAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_heads=4):
        super().__init__()

        # To project the decoder gate and the encoder's skip to a shared intermediate space (F_int)
        # Using the paper's semantics W_g, W_x
        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, 1, bias=True)

        # batch normalize both
        self.bn_g = nn.BatchNorm2d(F_int)
        self.bn_x = nn.BatchNorm2d(F_int)

        # Generate num_heads attention maps (multi-headed) todo refer to paper
        self.psi = nn.Conv2d(F_int, num_heads, 1, bias=True)
        # Merge heads to a single map todo refer to paper
        self.combine = nn.Conv2d(num_heads, 1, 1, bias=True)

    def forward(self, x, g):
        """
        Project the decoder gate and the encoder's skip to a shared intermediate space (F_int)
        :param x: from the encoder's skipped connection
        :param g: from the decoder gate position
        :return: (gated_skip, final_attn, head_maps)
        """
        x_int = self.bn_x(self.W_x(x))  # project to F_int and batch norm

        g_int = self.bn_g(self.W_g(g))  # project to F_int and batch norm
        # interpolate the gate from the decoder using bilinear
        g_int = functional.interpolate(g_int, size=x_int.shape[2:], mode='bilinear', align_corners=False)

        # Find alpha as an additive + ReLU
        alpha = (g_int + x_int).relu()

        # Get head maps and apply sigmoid for [0,1]
        heads = self.psi(alpha).sigmoid()
        heads = heads.sigmoid()

        # Combine to a single head map and apply sigmoid
        attn = self.combine(heads)
        attn = attn.sigmoid()

        # Generate the gated skip using the found attention
        gated_skip = x * attn

        return gated_skip, attn, heads
