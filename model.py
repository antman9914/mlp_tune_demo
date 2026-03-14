import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """两层 MLP：Linear → Activation → Dropout → Linear"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout_rate: float = 0.0, activation: str = "relu"):
        super().__init__()
        act_map = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_map.get(activation, nn.ReLU()),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
