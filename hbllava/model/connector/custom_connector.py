import torch
from torch import nn


class Custom_Connector(nn.Module):
    def __init__(self,input_channels=1024,output_channels=896):
        super().__init__()
        
        self.pre_norm = nn.LayerNorm(input_channels)
        self.proj = nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.GELU(),
            nn.Linear(input_channels, output_channels)
        )
        self.pool = nn.AvgPool1d(kernel_size=18, stride=18)  # 下采样序列长度

    def forward(self, x):
        x = self.pre_norm(x)
        x=  self.proj(x) # [batch, frames*576, output_channels]

        x = x.permute(0, 2, 1)      # 调整为 [batch, output_channels, frames*576] 以适应池化层
        x = self.pool(x)             # 输出形状: [batch, output_channels, 512]
        x = x.permute(0, 2, 1)      # 恢复为 [batch, 512, output_channels]
        return x