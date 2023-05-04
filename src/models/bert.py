import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# read the config file
with open('../config/config_v1.yaml') as f:
    config = yaml.safe_load(f)


class BertModel(nn.Module):
    def __init__(self) -> None:
        super(BertModel, self).__init__()


class SelfAttention(nn.Module):
    '''Implemented the Scaled Dot Product Attention proposed in Attention is All You Need
    by Vaswani et. al (page 4, section 3.2)

    Args:
    :param: `model_dimension`: int -> represents the embedding dimension which the transformer will handle
    :param: `embedding_dim`: int -> represents the  dimension of the input vector, from the nn.Embedding layer
    '''
    def __init__(self, head_dim: int, embedding_dim: int) -> None:
        super(SelfAttention, self).__init__()
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.q = nn.Linear(in_features=embedding_dim, out_features=head_dim) # find the projection of input for query
        self.k = nn.Linear(in_features=embedding_dim, out_features=head_dim) # find the projection of input for key
        self.v = nn.Linear(in_features=embedding_dim, out_features=head_dim) # find the projection of input for value

    @staticmethod
    def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(torch.bmm(Q, K.mT) / math.sqrt(Q.shape[-1]), dim=-1)
        outputs = torch.bmm(weights, V)
        return weights, outputs
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Q, K, V = self.q(x), self.k(x), self.v(x)
        weights, outputs = self.scaled_dot_product_attention(Q, K, V)
        return weights, outputs


class MultiHeadAttention(nn.Module):
    '''Implements the Multi-head attention as mentioned in the Attention is All you Need by Vaswani et. al.
    
    Args:
    :param: `model_dimension`: int -> represents the embedding dimension which the transformer will handle
    :param: `embedding_dim`: int -> represents the  dimension of the input vector, from the nn.Embedding layer
    :param: `num_heads`: int -> represents the number of heads needed for the training

    How does this work?
    -------------------
        * So, what we want to do is, find projections of the input embedding vector to the queries, keys and the values.
        * There are multiple implementations for the same, one which could be to reshape the tensors and implement self attention on top of that, which is more efficient 
          implementation, but for more readability and understanding, what we can do is the following:
            * We make the model learn the projects and map it to another dimension which is 
    '''
    def __init__(self, model_dimension: int, embedding_dim: int, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()
        self.model_dimension = model_dimension
        self.embedding_dim = self.embedding_dim
        self.num_heads = num_heads
        assert self.embedding_dim % self.num_heads == 0, "Make sure that the, number of heads is divisible by the embedding size"
        self.head_dim  = self.model_dimension // self.num_heads

        self.heads = nn.ModuleList([
            SelfAttention(embedding_dim=embedding_dim, head_dim=self.head_dim) for _ in range(self.num_heads)
        ])

        self.linear = nn.Linear(in_features=self.model_dimension, out_features=self.model_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.tensor([])
        weights = torch.tensor([])

        for head in self.heads:
            weight, output = head(x)
            outputs = torch.cat([outputs, output])
            weights = torch.cat([weights, weight])
        
        outputs = self.linear(outputs)

        return weights, outputs


class FeedForwardNetwork(nn.Module):
    def __init__(self, model_dimension: int, hidden_size: int) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(in_features=model_dimension, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=model_dimension)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        self.position_encoding_matrix = torch.zeros((max_len, d_model))
        position_ids = torch.arange(max_len).unsqueeze(1)
        dimension_ids = torch.arange(d_model)
        
        self.position_encoding_matrix = torch.where(
            dimension_ids % 2 == 0,
            torch.sin(position_ids / 10000**(2*dimension_ids / d_model)), 
            torch.cos(position_ids / 10000**(2*dimension_ids / d_model)), 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_encoding_matrix[:x.shape[1]] # Take sequence length

