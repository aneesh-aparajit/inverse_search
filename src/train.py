from typing import Dict, Optional, Any

import mlflow
import seaborn as sns
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.bert import BertModel
from src.models.clip import ClipModel
from src.models.vit import ViTModel
from src.utils import get_cnn, get_lm, get_optimizer, get_scheduler

with open("./config/cfg1.yaml") as f:
    config = yaml.safe_load(f)


def train_one_epoch(
    model: ClipModel,
    optimizer: th.optim,
    dataloader: DataLoader,
    scheduler: th.optim.lr_scheduler = None,
) -> Dict[str, Any[float, th.Tensor]]:
    model = model.train()

    dataset_size = 0.0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"(train) ")
    for step, batch in pbar:
        batch = {k: v.to(config["DEVICE"]) for k, v in batch.items()}
        batch_size = batch["images"].shape[0]
        outputs = model.forward(batch=batch)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += batch_size * loss.item()
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        mlflow.log_metric("train_step_loss", loss.item())

        pbar.set_postfix(loss=f"{epoch_loss:.5f}", step=step)

    return {
        "epoch_loss": epoch_loss,
        "last_loss": loss.item(),
        "similarity": outputs["similarity_matrix"],
    }

