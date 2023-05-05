from typing import Tuple
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import yaml

from src.models.bert import BertModel
from src.models.vit import ViTModel

with open("./config/cfg1.yaml") as f:
    config = yaml.safe_load(f)


def get_cnn():
    if config["HYPERPARAMETERS"]["CNN"] == "ViT":
        return ViTModel(config=config)
    raise NotImplementedError("Just ViT is implemented as of now...")


def get_lm():
    if config["HYPERPARAMETERS"]["LM"] == "BERT":
        return BertModel(config=config)
    raise NotImplementedError("Just BERT is implemented as of now...")


def optimizer(cnn: nn.Module, lm: nn.Module) -> Tuple[any, any]:
    cnn_optim = optim.Adam(
        params=cnn.parameters(), lr=config["HYPERPARMATERS"]["LEARNING_RATE"]
    )
    lm_optim = optim.Adam(
        params=lm.parameters(), lr=config["HYPERPARMATERS"]["LEARNING_RATE"]
    )
    return cnn_optim, lm_optim


def fetch_scheduler(optimizer):
    if config["HYPERPARMATERS"].scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["HYPERPARMATERS"]["T_MAX"],
            eta_min=config["HYPERPARMATERS"]["MIN_LR"],
        )
    elif config["HYPERPARMATERS"].scheduler == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config["HYPERPARMATERS"]["T_)"],
            eta_min=config["HYPERPARMATERS"]["MIN_LR"],
        )
    elif config["HYPERPARMATERS"].scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=7,
            threshold=0.0001,
            min_lr=config["HYPERPARMATERS"]["MIN_LR"],
        )
    elif config["HYPERPARMATERS"].scheduler == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif config["HYPERPARMATERS"].scheduler == None:
        return None

    return scheduler
