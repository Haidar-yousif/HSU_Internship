import torch
import torch.nn as nn

from utils.loss.losses import SADLoss, GammaKL


def get_reconstruction_loss_fn(loss_name,reduction ='mean',mode='default'):

    if loss_name == "sad":
        loss_fn = SADLoss(reduction="sum",mode=mode,reduction=reduction)

    elif loss_name == "mse":
        loss_fn = nn.MSELoss(reduction="sum")

    elif loss_name == "cosine_similarity":
        loss_fn = nn.CosineSimilarity(dim=-1, eps=1e-8)

    elif loss_name == "bce":
        loss_fn = nn.BCELoss(reduction="sum")

    # TODO: add SID
    elif loss_name == "sid":
        raise ValueError("SID loss not available yet.")

    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return loss_fn



def get_regularization_loss_fn(concentration_sets, loss_name,reduction='mean',mode='default'):

    if loss_name == "gamma_div":
        loss_fn = GammaKL(alphas=torch.from_numpy(concentration_sets),reduction=reduction,mode=mode)

    else:
        raise ValueError(f"Unsupported regularization loss function: {loss_name}")

    return loss_fn

