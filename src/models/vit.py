import math
from typing import Dict, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dataclasses import dataclass

# read the config file
with open("../config/cfg1.yaml") as f:
    config = yaml.safe_load(f)


@dataclass
class ViTOutputClass:
    vit_output: th.Tensor
    vit_pooled_output: th.Tensor
    attention_weights: Dict[str, th.Tensor]


class ViTPatchEmbedding(nn.Module):
    """Implements the Patch Embedding on the image.

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
    """

    def __init__(
        self, embedding_dim: int = 768, patch_size: int = 16, in_channels: int = 3
    ) -> None:
        super(ViTPatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        self.layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.layer(x)  # (B, EMB, X, Y)
        B, N, X, Y = x.shape
        x = x.reshape(B, N, X * Y)  # (B, EMB, X*Y)
        x = x.permute(0, 2, 1)  # (B, X*Y, EMB)
        return x


class ViTSelfAttention(nn.Module):
    def __init__(
        self, head_dim: int, embedding_dim: int = 768, dropout: float = 0.1
    ) -> None:
        super(ViTSelfAttention, self).__init__()
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim

        self.query = nn.Linear(
            in_features=self.embedding_dim, out_features=self.head_dim
        )
        self.key = nn.Linear(in_features=self.embedding_dim, out_features=self.head_dim)
        self.value = nn.Linear(
            in_features=self.embedding_dim, out_features=self.head_dim
        )
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def scaled_dot_product_attention(
        query: th.Tensor, key: th.Tensor, value: th.Tensor
    ) -> Tuple[th.Tensor]:
        weights = F.softmax(th.bmm(query, key.mT) / math.sqrt(query.shape[-1]), dim=-1)
        outputs = th.bmm(weights, value)
        return weights, outputs

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        q, k, v = self.query(x), self.key(x), self.value(x)
        weights, outputs = self.scaled_dot_product_attention(query=q, key=k, value=v)
        return weights, self.dropout(outputs)


class ViTMultiHeadAttention(nn.Module):
    def __init__(
        self, embedding_dim: int = 768, num_heads: int = 8, dropout: float = 0.1
    ):
        super(ViTMultiHeadAttention, self).__init__()
        assert (
            embedding_dim % num_heads == 0
        ), "The number of heads must be a factor of the embedding dimensions"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads

        self.heads = nn.ModuleList(
            [
                ViTSelfAttention(
                    head_dim=self.head_dim, embedding_dim=self.embedding_dim
                )
                for _ in range(self.num_heads)
            ]
        )
        self.output = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        weights, outputs = th.tensor([]), th.tensor([])

        for head in self.heads:
            weight, output = head(x)
            weights = th.cat([weights, weight], dim=-1)
            outputs = th.cat([outputs, output], dim=-1)

        return weights, self.dropout(self.output(outputs))


class ViTPositionEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        max_token_length: int = 500,
        dropout: float = 0.1,
    ) -> None:
        super(ViTPositionEmbedding, self).__init__()
        self.position_encoding_matrix = th.zeros((max_token_length, embedding_dim))
        position_ids = th.arange(max_token_length).unsqueeze(1)
        dimension_ids = th.arange(embedding_dim)

        self.position_encoding_matrix = th.where(
            dimension_ids % 2 == 0,
            th.sin(position_ids / 10000 ** (2 * dimension_ids / embedding_dim)),
            th.cos(position_ids / 10000 ** (2 * dimension_ids / embedding_dim)),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.dropout(
            x + self.position_encoding_matrix[: x.shape[1]]
        )  # Take sequence length


class ViTMlpLayer(nn.Module):
    def __init__(
        self, embedding_dim: int = 768, hidden_dim: int = 3072, dropout: float = 0.1
    ) -> None:
        super(ViTMlpLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.mlp_activation_fn = nn.GELU()
        self.linear_1 = nn.Linear(
            in_features=embedding_dim, out_features=hidden_dim, bias=True
        )
        self.linear_2 = nn.Linear(
            in_features=hidden_dim, out_features=embedding_dim, bias=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.mlp_activation_fn(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class ViTTransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 8,
        hidden_dim: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super(ViTTransformerEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mha = ViTMultiHeadAttention(
            embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout
        )
        self.layer_norm_1 = nn.LayerNorm(
            normalized_shape=(embedding_dim,), eps=1e-12, elementwise_affine=True
        )
        self.layer_norm_2 = nn.LayerNorm(
            normalized_shape=(embedding_dim,), eps=1e-12, elementwise_affine=True
        )
        self.mlp = ViTMlpLayer(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        x_ = x.clone()
        weights, x = self.mha(x)
        x = self.layer_norm_1(x) + x_
        x_ = x.clone()
        x = self.mlp(x)
        x = self.layer_norm_2(x) + x_
        return weights, x


class ViTModel(nn.Module):
    def __init__(self, config: Dict[str, any]) -> None:
        super(ViTModel, self).__init__()
        self.config = config
        self.patch_embed = ViTPatchEmbedding(
            embedding_dim=config["ViT"]["BASELINE_CONFIG"]["EMBEDDING_DIM"],
            patch_size=config["ViT"]["BASELINE_CONFIG"]["PATCH_SIZE"],
            in_channels=config["ViT"]["BASELINE_CONFIG"]["NUM_CHANNELS"],
        )
        self.positional_embed = ViTPositionEmbedding(
            embedding_dim=config["ViT"]["BASELINE_CONFIG"]["EMBEDDING_DIM"],
            max_token_length=config["ViT"]["BASELINE_CONFIG"]["MAX_TOKEN_SIZE"],
            dropout=config["ViT"]["BASELINE_CONFIG"]["DROPOUT"],
        )

        self.encoder = nn.ModuleList(
            [
                ViTTransformerEncoder(
                    embedding_dim=config["ViT"]["BASELINE_CONFIG"]["EMBEDDING_DIM"],
                    num_heads=config["ViT"]["BASELINE_CONFIG"]["NUM_HEADS"],
                    hidden_dim=config["ViT"]["BASELINE_CONFIG"]["HIDDEN_DIM"],
                    dropout=config["ViT"]["BASELINE_CONFIG"]["DROPOUT"],
                )
                for _ in range(config["ViT"]["BASELINE_CONFIG"]["ENCODER_LAYERS"])
            ]
        )

        self.pooler = ViTPooler(
            in_features=config["ViT"]["BASELINE_CONFIG"]["EMBEDDING_DIM"],
            out_features=config["ViT"]["BASELINE_CONFIG"]["EMBEDDING_DIM"],
        )

    def forward(self, x: th.Tensor) -> ViTOutputClass:
        x_embedding = self.patch_embed.forward(x=x)
        # print(x_embedding.shape)
        x = x_embedding + self.positional_embed.forward(x=x_embedding)
        weights = {}
        for layer_num, layer in enumerate(self.encoder):
            wei, x = layer(x)
            weights[f"layer_{layer_num}"] = wei
        vit_output, vit_pooled_output = self.pooler(x)
        return ViTOutputClass(
            vit_output=vit_output,
            vit_pooled_output=vit_pooled_output,
            attention_weights=weights,
        )


class ViTPooler(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(ViTPooler, self).__init__()
        self.dense = nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )
        self.activation = nn.Tanh()

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        bert_output = self.activation(self.dense(x))
        bert_pooled_output = bert_output.mean(axis=1)
        return bert_output, bert_pooled_output


if __name__ == "__main__":
    m = ViTModel(config=config)
    x = th.randn(size=(32, 3, 224, 224))
    y = m(x)
    print(
        f"""x: {x.shape}
vit_output: {y.vit_output.shape}
vit_pooled_output: {y.vit_pooled_output.shape}"""
    )
