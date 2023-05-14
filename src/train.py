import mlflow
import torch as th
from torch.cuda import amp
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
    optimizer,
    dataloader: DataLoader,
    epoch: int,
    scheduler=None,
) -> float:
    model.train()
    scaler = th.cuda.amp.grad_scaler.GradScaler()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="(train) ")

    # with mlflow.start_run(run_name=f"ClipModel-epoch-{epoch}"):
    dataset_size = 0.0
    running_size = 0.0

    for step, batch in pbar:
        batch_size = batch["image"].shape[0]

        with amp.autocast_mode.autocast():
            outputs = model.forward(batch=batch)

        scaler.scale(outputs=outputs["loss"]).backward()
        scaler.step(optimizer=optimizer)
        scaler.update()

        running_size += outputs["loss"].item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_size / dataset_size

        if scheduler is not None:
            scheduler.step()

        pbar.set_postfix(loss=f"{epoch_loss:>.5f}")

        mlflow.log_metric("train_loss", epoch_loss, step=step)

    return epoch_loss


@th.no_grad()
def valid_one_epoch(model: ClipModel, dataloader: DataLoader, epoch: int) -> float:
    model.eval()
    dataset_size = 0.0
    running_size = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="(valid) ")

    for step, batch in pbar:
        batch_size = batch["image"].shape[0]

        with amp.autocast_mode.autocast():
            outputs = model.forward(batch=batch)

        running_size += outputs["loss"].item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_size / dataset_size

        pbar.set_postfix(loss=f"{epoch_loss:>.5f}")

        mlflow.log_metric("valid_loss", epoch_loss, step=step)

    return epoch_loss
