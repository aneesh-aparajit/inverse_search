import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dataclasses import dataclass, fields

# read the config file
with open("../config/config_v1.yaml") as f:
    config = yaml.safe_load(f)


@dataclass
class BertOutputDataClass:
    bert_output: torch.Tensor
    bert_pooled_output: torch.Tensor
    attention_weights: torch.Tensor


class BertModel(nn.Module):
    def __init__(self, config: Dict[str, any]) -> None:
        super(BertModel, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config["LANGUAGE"]["BASELINE_CONFIG"]["NUM_EMBEDDINGS"],
            embedding_dim=config["LANGUAGE"]["BASELINE_CONFIG"]["EMBEDDING_DIM"],
        )
        self.position_embeddings = BertEmbeddings(
            model_dimension=config["LANGUAGE"]["BASELINE_CONFIG"]["MODEL_DIMENSION"]
        )

        self.encoder = BertEncoder(
            model_dimension=config["LANGUAGE"]["BASELINE_CONFIG"]["MODEL_DIMENSION"],
            embedding_dim=config["LANGUAGE"]["BASELINE_CONFIG"]["EMBEDDING_DIM"],
            hidden_size=config["LANGUAGE"]["BASELINE_CONFIG"]["HIDDEN_SIZE"],
            num_encoder_layers=config["LANGUAGE"]["BASELINE_CONFIG"][
                "NUM_ENCODER_LAYERS"
            ],
            intermediate_size=config["LANGUAGE"]["BASELINE_CONFIG"][
                "INTERMEDIATE_SIZE"
            ],
        )

        self.pooler = BertPooler(
            in_features=config["LANGUAGE"]["BASELINE_CONFIG"]["MODEL_DIMENSION"],
            out_features=config["LANGUAGE"]["BASELINE_CONFIG"]["MODEL_DIMENSION"],
        )

    def forward(self, x: torch.Tensor) -> BertOutputDataClass:
        x_embeddings = self.embedding(x)
        x_embeddings = self.position_embeddings(x_embeddings)
        weights, output = self.encoder.forward(x_embeddings)
        bert_output, bert_pooled_output = self.pooler(output)
        return BertOutputDataClass(
            bert_output=bert_output,
            bert_pooled_output=bert_pooled_output,
            attention_weights=weights,
        )


class BertPooler(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bert_output = self.activation(self.dense(x))
        bert_pooled_output = bert_output.mean(axis=1)
        return bert_output, bert_pooled_output


class BertSelfAttention(nn.Module):
    """Implemented the Scaled Dot Product Attention proposed in Attention is All You Need
    by Vaswani et. al (page 4, section 3.2)

    Args:
    :param: `model_dimension`: int -> represents the embedding dimension which the transformer will handle
    :param: `embedding_dim`: int -> represents the  dimension of the input vector, from the nn.Embedding layer
    """

    def __init__(self, head_dim: int, embedding_dim: int) -> None:
        super(BertSelfAttention, self).__init__()
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(
            in_features=embedding_dim, out_features=head_dim
        )  # find the projection of input for query
        self.key = nn.Linear(
            in_features=embedding_dim, out_features=head_dim
        )  # find the projection of input for key
        self.value = nn.Linear(
            in_features=embedding_dim, out_features=head_dim
        )  # find the projection of input for value
        self.dropout = nn.Dropout(p=config["LANGUAGE"]["BASELINE_CONFIG"]["DROPOUT"])

    @staticmethod
    def scaled_dot_product_attention(
        Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        weights = F.softmax(torch.bmm(Q, K.mT) / math.sqrt(Q.shape[-1]), dim=-1)
        outputs = torch.bmm(weights, V)
        return weights, outputs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Q, K, V = self.query(x), self.key(x), self.value(x)
        weights, outputs = self.scaled_dot_product_attention(Q, K, V)
        return weights, self.dropout(outputs)


class BertAttention(nn.Module):
    """Implements the Multi-head attention as mentioned in the Attention is All you Need by Vaswani et. al.

    Args:
    :param: `model_dimension`: int -> represents the embedding dimension which the transformer will handle
    :param: `embedding_dim`: int -> represents the  dimension of the input vector, from the nn.Embedding layer
    :param: `num_heads`: int -> represents the number of heads needed for the training

    How does this work?
    -------------------
        * So, what we want to do is, find projections of the input embedding vector to the queries, keys and the values.
        * There are multiple implementations for the same, one which could be to reshape the tensors and implement self attention
        on top of that, which is more efficient
        implementation, but for more readability and understanding, what we can do is the following:
            * We make the model learn the projects and map it to another dimension which is
    """

    def __init__(
        self, model_dimension: int, embedding_dim: int, num_heads: int = 8
    ) -> None:
        super(BertAttention, self).__init__()
        self.model_dimension = model_dimension
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert (
            self.embedding_dim % self.num_heads == 0
        ), "Make sure that the, number of heads is divisible by the embedding size"
        self.head_dim = self.model_dimension // self.num_heads

        self.heads = nn.ModuleList(
            [
                BertSelfAttention(
                    embedding_dim=self.embedding_dim, head_dim=self.head_dim
                )
                for _ in range(self.num_heads)
            ]
        )

        self.output = BertSelfOutput(
            in_features=self.model_dimension, out_features=self.model_dimension
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.tensor([])
        weights = torch.tensor([])

        for head in self.heads:
            weight, output = head(x)
            outputs = torch.cat([outputs, output], dim=-1)
            weights = torch.cat([weights, weight], dim=-1)
        
        outputs = self.output(outputs)

        return weights, outputs


class BertSelfOutput(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )
        self.LayerNorm = nn.LayerNorm(
            normalized_shape=(out_features,), eps=1e-12, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=config["LANGUAGE"]["BASELINE_CONFIG"]["DROPOUT"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.LayerNorm(self.dense(x)))


class BertIntermediate(nn.Module):
    def __init__(self, in_features: int, intermediate_size: int) -> None:
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(in_features=in_features, out_features=intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(x))


class BertOutput(nn.Module):
    """Implements the FFN proposed by Vaswani Et. al in Attention is All you Need (page 5, section 3.3)

    Args:
    :param: `model_dimension`: int -> represents the embedding dimension which the transformer will handle
    :param: `embedding_dim`: int -> represents the  dimension of the input vector, from the nn.Embedding layer
    """

    def __init__(self, model_dimension: int, hidden_size: int) -> None:
        super(BertOutput, self).__init__()
        self.linear1 = nn.Linear(in_features=model_dimension, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=model_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.linear1(x))
        return self.linear2(x)


class BertEmbeddings(nn.Module):
    def __init__(self, model_dimension: int = 512, max_token_length: int = 5000):
        super(BertEmbeddings, self).__init__()

        self.position_encoding_matrix = torch.zeros((max_token_length, model_dimension))
        position_ids = torch.arange(max_token_length).unsqueeze(1)
        dimension_ids = torch.arange(model_dimension)

        self.position_encoding_matrix = torch.where(
            dimension_ids % 2 == 0,
            torch.sin(position_ids / 10000 ** (2 * dimension_ids / model_dimension)),
            torch.cos(position_ids / 10000 ** (2 * dimension_ids / model_dimension)),
        )
        self.dropout = nn.Dropout(p=config["LANGUAGE"]["BASELINE_CONFIG"]["DROPOUT"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            x + self.position_encoding_matrix[: x.shape[1]]
        )  # Take sequence length


class BertLayer(nn.Module):
    def __init__(
        self, model_dimension: int, embedding_dim: int, hidden_size: int
    ) -> None:
        super(BertLayer, self).__init__()
        self.model_dimension = model_dimension
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.layernorm1 = nn.LayerNorm(normalized_shape=self.model_dimension)
        self.layernorm2 = nn.LayerNorm(normalized_shape=self.model_dimension)

        self.mha = BertAttention(
            model_dimension=self.model_dimension, embedding_dim=self.embedding_dim
        )
        self.output = BertOutput(
            model_dimension=self.model_dimension, hidden_size=self.hidden_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # mha
        x_ = x.clone()
        weights, x = self.mha(x)
        x = x_ + self.layernorm1(x)
        # ffn
        x_ = x.clone()
        x = self.output(x)
        x = x_ + self.layernorm2(x)

        return weights, x


class BertEncoder(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        embedding_dim: int,
        intermediate_size: int,
        hidden_size: int,
        num_encoder_layers: int,
    ) -> None:
        super(BertEncoder, self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.model_dimension = model_dimension
        self.hidden_size = hidden_size
        self.embedding_size = embedding_dim
        self.intermediate_size = intermediate_size

        self.layers = nn.ModuleList(
            [
                BertLayer(
                    model_dimension=self.model_dimension,
                    embedding_dim=self.embedding_size,
                    hidden_size=self.hidden_size,
                )
                for _ in range(self.num_encoder_layers)
            ]
        )

        self.intermediate = BertIntermediate(
            in_features=self.model_dimension,
            intermediate_size=self.intermediate_size,
        )

        self.output = BertSelfOutput(
            in_features=self.intermediate_size, out_features=self.hidden_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        weights = {}
        for layer_num, layer in enumerate(self.layers):
            wei, x = layer(x)
            weights[f"layer_{layer_num}"] = wei
        x = self.intermediate(x)
        # print(x.shape, self.output.parameters)
        x = self.output(x)
        return weights, x


if __name__ == "__main__":
    model = BertModel(config=config)
    inputs = torch.randint(
        low=0,
        high=config["LANGUAGE"]["BASELINE_CONFIG"]["NUM_EMBEDDINGS"],
        size=(32, 128),
    )
    outputs = model.forward(inputs)

    print(model)

    print('Output Dataclass:-')
    print('---------')
    for field in fields(BertOutputDataClass):
        print(field.name)

    print(f'Input shape: {inputs.shape}')
    print(f'Bert Output: {outputs.bert_output.shape}')
    print(f'Bert Pooled Output: {outputs.bert_pooled_output.shape}')
    __import__('pprint').pprint({
        k: v.shape for k, v in outputs.attention_weights.items()
    })
