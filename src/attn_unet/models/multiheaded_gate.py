import torch.nn as nn
import torch.nn.functional as functional

'''
updated version of the multi-headed attention gate with:
1. softmax over all heads to compete over alignment pixels
2. added dropout
3. adding regularization to make heads differ? 
'''


class MultiHeadAttentionGate(nn.Module):
    def __init__(self,
                 F_g, F_l, F_int,
                 num_heads=4,
                 tau=1.0,
                 drop_prob=0.1
                 ):
        super().__init__()

        # To project the decoder gate and the encoder's skip to a shared intermediate space (F_int)
        # Using the paper's semantics W_g, W_x
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=True)

        # To batch normalize both
        self.bn_g = nn.BatchNorm2d(F_int)
        self.bn_x = nn.BatchNorm2d(F_int)

        # Generate num_heads attention maps (multi-headed)
        self.psi = nn.Conv2d(F_int, num_heads, kernel_size=1, bias=True)

        # Initialize heads as orthogonal todo
        nn.init.orthogonal_(self.psi.weight.view(self.psi.out_channels, -1))

        # Head dropout
        if drop_prob > 0:
            self.head_dropout = nn.Dropout2d(p=drop_prob)
        else:
            self.head_dropout = nn.Identity()

        # Merge heads to a single map
        self.combine = nn.Conv2d(num_heads, 1, 1, bias=True)

    def forward(self, x, g):
        """
        Project the decoder gate and the encoder's skip to a shared intermediate space (F_int)
        :param x: from the encoder's skipped connection
        :param g: from the decoder gate position
        :return: (gated_skip, final_attn, head_maps)
        """

        # project to F_int and batch norm
        x_int = self.bn_x(self.W_x(x))

        # project gate to F_int and batch norm & interpolate
        g_int = self.bn_g(self.W_g(g))
        g_int = functional.interpolate(g_int, size=x_int.shape[2:], mode='bilinear', align_corners=False)

        # Find alpha as an additive + ReLU
        alpha = (g_int + x_int).relu()

        # Get head logits and apply dropout
        heads_logits = self.psi(alpha)
        heads_logits = self.head_dropout(heads_logits)

        # Apply softmax to get competition between heads for each pixel
        heads = (heads_logits / max(self.tau, 1e-6)).softmax(dim=1)

        # Combine to a single head map and apply sigmoid
        attn = self.combine(heads)
        attn = attn.sigmoid()

        # Generate the gated skip using the found attention
        gated_skip = x * attn

        return gated_skip, attn, heads
