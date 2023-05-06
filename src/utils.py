from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler

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


def get_optimizer(model: nn.Module):
    """
    Returns the optimizer based on the config files.
    """
    if config["HYPERPARAMETERS"]["OPTIMIZER"] == "Adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=config["HYPERPARAMETERS"]["LEARNING_RATE"],
            rho=config["HYPERPARAMETERS"]["RHO"],
            eps=config["HYPERPARAMETERS"]["EPS"],
        )
    elif config["HYPERPARAMETERS"]["OPTIMIZER"] == "Adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=config["HYPERPARAMETERS"]["LEARNING_RATE"],
            lr_decay=config["HYPERPARAMETERS"]["LR_DECAY"],
            weight_decay=config["HYPERPARAMETERS"]["WEIGHT_DECAY"],
        )
    elif config["HYPERPARAMETERS"]["OPTIMIZER"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["HYPERPARAMETERS"]["LEARNING_RATE"],
            betas=config["HYPERPARAMETERS"]["BETAS"],
            eps=config["HYPERPARAMETERS"]["EPS"],
        )
    elif config["HYPERPARAMETERS"]["OPTIMIZER"] == "RMSProp":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config["HYPERPARAMETERS"]["LEARNING_RATE"],
            alpha=config["HYPERPARAMETERS"]["ALPHA"],
            eps=config["HYPERPARAMETERS"]["EPS"],
            weight_decay=config["HYPERPARAMETERS"]["WEIGHT_DECAY"],
            momentum=config["HYPERPARAMETERS"]["MOMENTUM"],
        )
    else:
        raise NotImplementedError(
            f"The optimizer {config.optimizer} has not been implemented."
        )
    return optimizer


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
