import torch
import torch.nn as nn


class MLP(nn.Module):
    """可配置深度的 MLP：[Linear → Activation → Dropout] × n_hidden_layers → Linear

    Args:
        input_dim:    输入维度
        hidden_dims:  每个隐藏层的维度列表，长度即隐藏层数
        output_dim:   输出维度（类别数）
        dropout_rate: dropout 概率，应用于每个隐藏层之后
        activation:   激活函数名称：relu / tanh / gelu
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 dropout_rate: float = 0.0, activation: str = "relu"):
        super().__init__()
        act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        act_cls = act_map.get(activation, nn.ReLU)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                act_cls(),
                nn.Dropout(p=dropout_rate),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
