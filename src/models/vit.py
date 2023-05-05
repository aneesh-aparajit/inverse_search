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

    What we are supposed to do is, split the images into patches and then find a embedding for each of them, so 
    instead what we can do is, combine both the steps together and run a convolution on the image of kernel_size
    as the patch_size and the stride also as the patch_size.

    We can map this to an out_channels of embedding_size, so each pixel will have a new representation for 
    embdding_size and would result in the size of (B, EMB, X, Y). Now, we reshape this to (B, X*Y, EMB).

    By doing this, the SEQUENCE_LENGTH becomes X*Y just like what we did in Attention is All You Need by 
    Vaswani Et. al.

    Inputs: 
    :param: `embedding_dim` -> the dimension for the embeddings, remains the same for the entire model also the output dim
    :param: `patch_size` -> the size at which you want the representation.
    :param: `in_channels` -> the number of channels in the input.
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


class ViTSelfAttention(nn.Module):
    def __init__(self, head_dim: int, embedding_dim: int = 768, dropout: float = 0.1) -> None:
        super(ViTSelfAttention, self).__init__()
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        
        self.query = nn.Linear(in_features=self.embedding_dim, out_features=self.head_dim)
        self.key = nn.Linear(in_features=self.embedding_dim, out_features=self.head_dim)
        self.value = nn.Linear(in_features=self.embedding_dim, out_features=self.head_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    @staticmethod
    def scaled_dot_product_attention(query: th.Tensor, key: th.Tensor, value: th.Tensor) -> Tuple[th.Tensor]:
        weights = F.softmax(th.bmm(query, key.mT) / math.sqrt(query.shape[-1]), dim=-1)
        outputs = th.bmm(weights, value)
        return weights, outputs

    def forward(self, x: th.Tensor) -> th.Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)
        return self.dropout(self.scaled_dot_product_attention(query=q, key=k, value=v))


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super(ViTMultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "The number of heads must be a factor of the embedding dimensions"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads

        self.norm = nn.LayerNorm(normalized_shape=(self.embedding_dim, ), eps=1e-12, elementwise_affine=True)
        self.heads = nn.ModuleList([
            ViTSelfAttention(head_dim=self.head_dim, embedding_dim=self.embedding_dim) for _ in range(self.num_heads)
        ])
        self.output = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        weights, outputs = th.tensor([]), th.tensor([])
        x = self.norm(x)

        for head in self.heads:
            weight, output = head(x)
            weights = th.cat([weights, weight], dim=-1)
            outputs = th.cat([outputs, output], dim=-1)

        return self.dropout(self.output(outputs))


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
