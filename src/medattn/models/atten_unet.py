import torch
import torch.nn as nn
from multiheaded_gate import MultiHeadAttentionGate


class ConvBlock(nn.Module):
    """
    Convolution block to be used in the classic U-Net structure
    with # channels in and out determined by c_in & c_out
    kernel: hard-coded 3x3

    seq: U-Net based with batch regularisation
    conv (c_in -> c_out) -> batch norm -> ReLU -> conv (c_out -> c_out) -> batch norm -> ReLU

    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture including Stacks2D for data and multiheaded gates
    32 base channels instead of 64 for Poc todo verify from paper
    explicitly handling the head number for the gates: Attention U-Net uses a single additive gate
    """
    def __init__(self,
                 in_channels=1, out_channels=1,
                 base_ch=32, num_heads=1):
        super().__init__()
        c = base_ch

        """
        Encoder parts, with channels doubling at every down step
        Encoder1 (ConvBlock) -> Encoder2 (..) -> Encoder3 -> Encoder3 -> Pooling layer -> Final Conv at cender level ("bottom" of U-Net)
        """
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, 2*c)
        self.enc3 = ConvBlock(2*c, 4*c)
        self.enc4 = ConvBlock(4*c, 8*c)
        self.pool = nn.MaxPool2d(2)
        self.cender = ConvBlock(8 * c, 16 * c)  # spatial [H/16, W/16]

        """
        Decoder parts, dec4 until dec1
        Same steps as the Attention U-Net, with MultiHeaded Attention
        each step: (no gate at last step to simplify) todo check paper and OG code
            up-sampling (factor of 2) ->
            attention gate (todo explain) F_g = gate channels (decoder), F_l skip channels (from encoder), F_int (reduced space for additive attention)-> 
            conv (uses the concatenated [upsamp_decoder, gated_skip_connection], so F_g + F_l and reduce by a factor of 2)
        final step: final output Conv with 1x1 kernel
        """

        self.up4 = nn.ConvTranspose2d(16*c, 8*c, 2, stride=2)
        self.gate4 = MultiHeadAttentionGate(F_g=8*c, F_l=8*c, F_int=4*c, num_heads=num_heads)
        self.dec4 = ConvBlock(16*c, 8*c)

        self.up3 = nn.ConvTranspose2d(8*c, 4*c, 2, stride=2)
        self.gate3 = MultiHeadAttentionGate(F_g=4*c, F_l=4*c, F_int=2*c, num_heads=num_heads)
        self.dec3 = ConvBlock(8*c, 4*c)

        self.up2 = nn.ConvTranspose2d(4*c, 2*c, 2, stride=2)
        self.gate2 = MultiHeadAttentionGate(F_g=2*c, F_l=2*c, F_int=c, num_heads=num_heads)
        self.dec2 = ConvBlock(4*c, 2*c)

        self.up1 = nn.ConvTranspose2d(2*c, c, 2, stride=2)
        self.dec1 = ConvBlock(2*c, c)

        self.out_conv = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        """
        Implement forward pass using the structure in the class init
        :param x:
        :return:
        """

        # Encoding with pooling between layers until the U-Net cender
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        center = self.cender(self.pool(e4))

        # Decoding using gates for matching U-Net steps
        # upsample, gate the same levels and concat
        d4 = self.up4(center)
        e4g, _, _ = self.gate4(e4, d4)  # (gated_skip, attn_map, head_maps)
        d4 = self.dec4(torch.cat([d4, e4g], 1))

        d3 = self.up3(d4)
        e3g, _, _ = self.gate3(e3, d3)
        d3 = self.dec3(torch.cat([d3, e3g], 1))

        d2 = self.up2(d3)
        e2g, _, _ = self.gate2(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2g], 1))

        # Skipping the final gate for PoC
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], 1))

        logits = self.out_conv(d1)

        return logits
