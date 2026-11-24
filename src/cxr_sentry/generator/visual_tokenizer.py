import torch
from torch import nn

class VisualTokenizer(nn.Module):
    def __init__(self, image_channels: int, embedding_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.projection = nn.Linear(image_channels, embedding_dim)
        self.positions = nn.Parameter(torch.zeros(1, 16, embedding_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        grid = self.pool(features)
        tokens = grid.flatten(2).transpose(1, 2)
        return self.projection(tokens) + self.positions

if __name__ == "__main__":
    fake_features = torch.randn(2, 1024, 7, 7)
    tokenizer = VisualTokenizer(image_channels=1024, embedding_dim=512)
    tokens = tokenizer(fake_features)
    assert tokens.shape == (2, 16, 512)
    print("Visual Token Shape: ", tuple(tokens.shape))

