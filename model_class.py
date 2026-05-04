"""
model_class.py:  Self-contained LightweightMV2 definition.
"""
import torch
import torch.nn as nn

#1d inverted residual block (expands in middle)
class _InvertedResidual1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand=6):
        super().__init__()
        mid          = in_ch * expand #for depthwise conv to work in high dim space
        self.use_res = (stride == 1 and in_ch == out_ch) #residual bool
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, mid, 1), #pointwise mixing (mid x (in_ch x 1))
            nn.BatchNorm1d(mid), nn.ReLU6(True),
            nn.Conv1d(mid, mid, 3, stride, 1, groups=mid), #depthwise conv 3x1, p=1
            nn.BatchNorm1d(mid), nn.ReLU6(True),
            nn.Conv1d(mid, out_ch, 1), #pointwise projection
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x): 
        return (x + self.conv(x)) if self.use_res else self.conv(x)


class LightweightMV2(nn.Module):
    """
    Input: (B, T, F/Channels)
    Output: (B, num_classes) in raw logits
    """
    def __init__(self, input_size: int, num_classes: int = 20, dropout: float = 0.2):
        super().__init__()
        #(output_channels, stride, expand_ratio, num_blocks)
        cfg = [(32,1,1,1),(48,2,6,2),(64,1,6,3),(96,2,6,2),(128,1,6,1)] 
        #first layer, 32 channels
        layers = [nn.Sequential(
            nn.Conv1d(input_size, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU6(True))]
        in_ch = 32
        for out_ch, s, e, n in cfg:
            for i in range(n):
                layers.append(_InvertedResidual1D(
                    in_ch, out_ch, stride=s if i == 0 else 1, expand=e))
                in_ch = out_ch
        layers += [nn.Conv1d(in_ch, 256, 1), nn.BatchNorm1d(256), nn.ReLU6(True)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), #(B, 256, 1)
            nn.Flatten(), #(B, 256)
            nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, x, lengths=None, padding_mask=None):
        return self.head(self.features(x.permute(0, 2, 1)))
