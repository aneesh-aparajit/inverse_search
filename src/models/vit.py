import math
from typing import Dict, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dataclasses import dataclass, fields

# read the config file
with open("../config/cfg1.yaml") as f:
    config = yaml.safe_load(f)


@dataclass
class ViTOutputClass:
    pass


class ViTPatchEmbedding(nn.Module):
    '''Implements the Patch Embedding on the image.
    
    In the original implementation of ViT, they use learnable patch embeddings which essentially means we can use
    convolutions to map them to a layer.
    '''
    def __init__(self, embedding_dim: int = 768, patch_size: int = 16, in_channels: int = 3) -> None:
        super(ViTPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        self.layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size, 
            stride=self.patch_size,
            padding=0
        ) 
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.layer(x) # (B, EMB, X, Y)
        B, N, X, Y = x.shape
        x = x.reshape(B, N, X*Y) # (B, EMB, X*Y)
        x = x.permute(0, 2, 1) # (B, X*Y, EMB)
        return x


class ViTPositionEmbedding(nn.Module):
    pass


class ViTMlpLayer(nn.Module):
    pass


class ViTModel(nn.Module):
    pass


if __name__ == '__main__':
    m = ViTPatchEmbedding()
    x = th.randn(size=(32, 3, 224, 224))
    y = m(x)
    print(f'x: {x.shape}\ny: {y.shape}')
