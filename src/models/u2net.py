"""
U²-Net architecture (Qin et al., Pattern Recognition 2020).
arXiv: https://arxiv.org/abs/2005.09007

Two-level nested U-structure: each block inside the U-Net backbone is itself
a small encoder-decoder (RSU block), giving richer multi-scale context with
fewer parameters than a plain deep U-Net.

Building blocks
---------------
REBNCONV  — single conv + BN + ReLU
RSU-N     — Residual U-block with N encoder/decoder levels + pooling
RSU-4F    — Like RSU-4 but replaces 3×3 conv with 3×3 dilated conv to avoid
             spatial collapse on already-small feature maps

Full U²-Net encoder/decoder follows the original paper exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class REBNCONV(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 dirate: int = 1, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=3, stride=stride,
                      padding=1 * dirate, dilation=dirate, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def _upsample_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Bilinear upsample `src` to spatial size of `ref`."""
    return F.interpolate(src, size=ref.shape[2:], mode="bilinear",
                         align_corners=False)


# ---------------------------------------------------------------------------
# RSU blocks
# ---------------------------------------------------------------------------

class RSU7(nn.Module):
    """RSU-7: 7-level residual U-block."""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1  = REBNCONV(out_ch, mid_ch)
        self.pool1      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2  = REBNCONV(mid_ch, mid_ch)
        self.pool2      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3  = REBNCONV(mid_ch, mid_ch)
        self.pool3      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4  = REBNCONV(mid_ch, mid_ch)
        self.pool4      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5  = REBNCONV(mid_ch, mid_ch)
        self.pool5      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6  = REBNCONV(mid_ch, mid_ch)
        self.rebnconv7  = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hx  = x
        hxin = self.rebnconvin(hx)

        hx1  = self.rebnconv1(hxin)
        hx   = self.pool1(hx1)
        hx2  = self.rebnconv2(hx)
        hx   = self.pool2(hx2)
        hx3  = self.rebnconv3(hx)
        hx   = self.pool3(hx3)
        hx4  = self.rebnconv4(hx)
        hx   = self.pool4(hx4)
        hx5  = self.rebnconv5(hx)
        hx   = self.pool5(hx5)
        hx6  = self.rebnconv6(hx)
        hx7  = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat([hx7, hx6], 1))
        hx6d = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat([hx6d, hx5], 1))
        hx5d = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat([hx5d, hx4], 1))
        hx4d = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat([hx4d, hx3], 1))
        hx3d = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat([hx3d, hx2], 1))
        hx2d = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat([hx2d, hx1], 1))

        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1  = REBNCONV(out_ch, mid_ch)
        self.pool1      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2  = REBNCONV(mid_ch, mid_ch)
        self.pool2      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3  = REBNCONV(mid_ch, mid_ch)
        self.pool3      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4  = REBNCONV(mid_ch, mid_ch)
        self.pool4      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5  = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6  = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1  = self.rebnconv1(hxin)
        hx2  = self.rebnconv2(self.pool1(hx1))
        hx3  = self.rebnconv3(self.pool2(hx2))
        hx4  = self.rebnconv4(self.pool3(hx3))
        hx5  = self.rebnconv5(self.pool4(hx4))
        hx6  = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat([hx6, hx5], 1))
        hx5d = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat([hx5d, hx4], 1))
        hx4d = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat([hx4d, hx3], 1))
        hx3d = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat([hx3d, hx2], 1))
        hx2d = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat([hx2d, hx1], 1))

        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1  = REBNCONV(out_ch, mid_ch)
        self.pool1      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2  = REBNCONV(mid_ch, mid_ch)
        self.pool2      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3  = REBNCONV(mid_ch, mid_ch)
        self.pool3      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4  = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5  = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1  = self.rebnconv1(hxin)
        hx2  = self.rebnconv2(self.pool1(hx1))
        hx3  = self.rebnconv3(self.pool2(hx2))
        hx4  = self.rebnconv4(self.pool3(hx3))
        hx5  = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat([hx5, hx4], 1))
        hx4d = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat([hx4d, hx3], 1))
        hx3d = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat([hx3d, hx2], 1))
        hx2d = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat([hx2d, hx1], 1))

        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1  = REBNCONV(out_ch, mid_ch)
        self.pool1      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2  = REBNCONV(mid_ch, mid_ch)
        self.pool2      = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3  = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4  = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1  = self.rebnconv1(hxin)
        hx2  = self.rebnconv2(self.pool1(hx1))
        hx3  = self.rebnconv3(self.pool2(hx2))
        hx4  = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat([hx4, hx3], 1))
        hx3d = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat([hx3d, hx2], 1))
        hx2d = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat([hx2d, hx1], 1))

        return hx1d + hxin


class RSU4F(nn.Module):
    """RSU-4F: dilated convolutions instead of pooling (for small feature maps)."""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1  = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2  = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3  = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4  = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1  = self.rebnconv1(hxin)
        hx2  = self.rebnconv2(hx1)
        hx3  = self.rebnconv3(hx2)
        hx4  = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat([hx4, hx3], 1))
        hx2d = self.rebnconv2d(torch.cat([hx3d, hx2], 1))
        hx1d = self.rebnconv1d(torch.cat([hx2d, hx1], 1))

        return hx1d + hxin


# ---------------------------------------------------------------------------
# Full U²-Net
# ---------------------------------------------------------------------------

class U2NET(nn.Module):
    """U²-Net full model (large variant, ~44 M parameters).

    Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O. R., & Jagersand, M.
    (2020). U2-Net: Going Deeper with Nested U-Structure for Salient Object
    Detection. Pattern Recognition, 106, 107404.
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        # Encoder
        self.stage1     = RSU7(in_ch, 32, 64)
        self.pool12     = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2     = RSU6(64, 32, 128)
        self.pool23     = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3     = RSU5(128, 64, 256)
        self.pool34     = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4     = RSU4(256, 128, 512)
        self.pool45     = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5     = RSU4F(512, 256, 512)
        self.pool56     = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6     = RSU4F(512, 256, 512)

        # Decoder
        self.stage5d    = RSU4F(1024, 256, 512)
        self.stage4d    = RSU4(1024, 128, 256)
        self.stage3d    = RSU5(512, 64, 128)
        self.stage2d    = RSU6(256, 32, 64)
        self.stage1d    = RSU7(128, 16, 64)

        # Deep supervision side outputs (6 maps)
        self.side1      = nn.Conv2d(64,  out_ch, 3, padding=1)
        self.side2      = nn.Conv2d(64,  out_ch, 3, padding=1)
        self.side3      = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4      = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5      = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6      = nn.Conv2d(512, out_ch, 3, padding=1)

        # Fused output (concatenation of 6 side outputs → conv)
        self.outconv    = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[2], x.shape[3]

        # Encoder
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))

        # Decoder
        hx5d = self.stage5d(torch.cat([_upsample_like(hx6, hx5), hx5], 1))
        hx4d = self.stage4d(torch.cat([_upsample_like(hx5d, hx4), hx4], 1))
        hx3d = self.stage3d(torch.cat([_upsample_like(hx4d, hx3), hx3], 1))
        hx2d = self.stage2d(torch.cat([_upsample_like(hx3d, hx2), hx2], 1))
        hx1d = self.stage1d(torch.cat([_upsample_like(hx2d, hx1), hx1], 1))

        # Side outputs (upsampled to input resolution)
        d1 = F.interpolate(self.side1(hx1d), (h, w), mode="bilinear", align_corners=False)
        d2 = F.interpolate(self.side2(hx2d), (h, w), mode="bilinear", align_corners=False)
        d3 = F.interpolate(self.side3(hx3d), (h, w), mode="bilinear", align_corners=False)
        d4 = F.interpolate(self.side4(hx4d), (h, w), mode="bilinear", align_corners=False)
        d5 = F.interpolate(self.side5(hx5d), (h, w), mode="bilinear", align_corners=False)
        d6 = F.interpolate(self.side6(hx6),  (h, w), mode="bilinear", align_corners=False)

        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))

        return (
            torch.sigmoid(d0),
            torch.sigmoid(d1), torch.sigmoid(d2),
            torch.sigmoid(d3), torch.sigmoid(d4),
            torch.sigmoid(d5), torch.sigmoid(d6),
        )
