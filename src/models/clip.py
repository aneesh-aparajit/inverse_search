from typing import Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from src.models.bert import BertModel
from src.models.vit import ViTModel


class ClipModel(nn.Module):
    def __init__(
        self,
        cnn: ViTModel,
        lm: BertModel,
        cnn_embed_dim: int = 768,
        lm_embed_dim: int = 768,
        clip_embed_dim: int = 256,
    ) -> None:
        super(ClipModel, self).__init__()
        self.cnn = cnn
        self.lm = lm
        self.cnn_embed_dim = cnn_embed_dim
        self.lm_embed_dim = lm_embed_dim
        self.clip_embed_dim = clip_embed_dim

        self.cnn_linear = nn.Linear(
            in_features=self.cnn_embed_dim, out_features=self.clip_embed_dim, bias=True
        )
        self.lm_linear = nn.Linear(
            in_features=self.lm_embed_dim, out_features=self.clip_embed_dim, bias=True
        )

    def forward(self, batch: Dict[str, th.Tensor]):
        lm_embedding = self.lm.forward(batch["input_ids"]).bert_pooled_output
        cnn_embedding = self.cnn.forward(batch["image"]).vit_pooled_output

        lm_embedding = self.lm_linear(lm_embedding)
        cnn_embedding = self.cnn_linear(cnn_embedding)

        similarity_matrix = self.get_similarity_matrix(
            lm=lm_embedding, cnn=cnn_embedding
        )
        loss = self.get_loss(
            similarity_matrix=similarity_matrix, batch_size=batch["input_ids"].shape[0]
        )

        return {
            "lm_embedding": lm_embedding,
            "cnn_embedding": cnn_embedding,
            "similarity_matrix": similarity_matrix,
            "loss": loss,
        }

    @staticmethod
    def get_similarity_matrix(lm: th.Tensor, cnn: th.Tensor) -> th.Tensor:
        lm = F.normalize(input=lm, p=2)
        cnn = F.normalize(input=cnn, p=2)
        return lm @ cnn.mT

    @staticmethod
    def get_loss(similarity_matrix: th.Tensor, batch_size: int) -> th.Tensor:
        # get the labels
        labels = th.arange(0, batch_size, dtype=th.long)
        # find loss along the horizontal axis
        loss0 = F.cross_entropy(input=similarity_matrix, target=labels)
        # find loss along vertical axis
        loss1 = F.cross_entropy(input=similarity_matrix.mT, target=labels)
        return (loss0 + loss1) / 2


if __name__ == "__main__":
    import yaml

    with open("../config/cfg1.yaml") as f:
        config = yaml.safe_load(f)

    batch = {
        "input_ids": th.randint(
            low=0,
            high=config["BERT"]["BASELINE_CONFIG"]["NUM_EMBEDDINGS"],
            size=(32, 128),
        ),
        "image": th.randn(size=(32, 3, 224, 224)),
    }

    lm = BertModel(config=config)
    cnn = ViTModel(config=config)

    clip = ClipModel(cnn=cnn, lm=lm)

    outputs = clip.forward(batch=batch)

    print(outputs.keys())

    __import__("pprint").pprint({k: v.shape for k, v in outputs.items()})
