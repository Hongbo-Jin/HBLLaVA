import torch
from torch import nn


class Custom_Connector(nn.Module):
    def __init__(self,input_channels=1024,output_channels=2560):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels)  # 扩展特征维度
        self.pool = nn.AvgPool1d(kernel_size=18, stride=18)  # 下采样序列长度

    def forward(self, x):
        # x : [batch, seq_len=9216, features=input_channels]
        x = self.linear(x)          # 输出形状: [batch, 9216, output_channels]
        x = x.permute(0, 2, 1)      # 调整为 [batch, output_channels, 9216] 以适应池化层
        x = self.pool(x)             # 输出形状: [batch, output_channels, 512]
        x = x.permute(0, 2, 1)      # 恢复为 [batch, 512, output_channels]
        return x