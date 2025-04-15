import torch
import torch.nn as nn
import numpy as np
class SADLoss:
    """
        Spectral Angle Distance (SAD) loss function.

        Parameters
        ----------
        eps : avoid division by zero
        reduction : reduction method for the loss.
                    Options are "sum", "mean" or "none"
                            (the latter returns a tensor of size (..., n_samples)).
        mode : "default" for simple spectral vector as sample,"cnn" for patch images
        Returns
        -------
        out: shape (n_batch,
             spectral angle distance
    """
    def __init__(self,
                 reduction: str = "mean",
                 eps: float = 1e-8,mode:str ='default') -> None:

        self.reduction = reduction
        self.eps       = torch.tensor(eps)
        self.mode = mode

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction='{self.reduction}', eps={self.eps})"

    def to(self, device):
        self.eps.to(device)
        return self

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if len(target.shape) < 2: target = target.unsqueeze(0)
        if len(input.shape)  < 2: input  = input.unsqueeze(0)
        if self.mode == 'default':  # For 1D spectral vectors (B, C)
            target_norm = torch.norm(target, p=2, dim=1)
            input_norm  = torch.norm(input,  p=2, dim=1)
        else:  # For hyperspectral patches (B, C, H, W)
            target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
            input_norm  = torch.norm(input,  p=2, dim=1, keepdim=True)

        norm_factor = target_norm * input_norm

        scalar_product = torch.sum(target * input, dim=1)

        # eps at denominator + 1e-6 in cos for numerical stability
        cos = scalar_product / torch.max(norm_factor, self.eps)
        cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)

        if   self.reduction == "sum":  loss = torch.acos(cos).sum()
        elif self.reduction == "mean": loss = torch.acos(cos).mean()
        elif self.reduction == "none": loss = torch.acos(cos)
        else:
            raise ValueError("Invalid reduction type. Must be either'sum', 'none' or 'mean'.")

        return loss



class GammaKL:
    def __init__(self, alphas: torch.Tensor, reduction: str = "mean", mode: str = "default", eps: float = 1e-6):
        if mode not in ["default", "cnn"]:
            raise ValueError("mode must be 'default' or 'cnn'")
        self.alphas = alphas.to(dtype=torch.float32)
        self.reduction = reduction
        self.mode = mode
        self.eps = eps

    def to(self, device):
        self.alphas = self.alphas.to(device)
        return self

    def __call__(self, input: torch.Tensor):
        input = torch.clamp(input, min=self.eps, max=30.0)
        if self.mode == "default":
            batch_size = input.shape[0]
            alphas_prior = self.alphas.expand(batch_size, -1)
            sum_dim = -1
        else:
            batch_size, n_ems, H, W = input.shape
            alphas_prior = self.alphas.view(1, n_ems, 1, 1).expand(batch_size, n_ems, H, W)
            sum_dim = 1
        kl = (input - alphas_prior) * torch.digamma(input + self.eps) - \
             torch.lgamma(input + self.eps) + torch.lgamma(alphas_prior + self.eps)
        loss = torch.sum(kl, dim=sum_dim)
        if self.mode == "cnn" and self.reduction in ["sum", "mean"]:
            loss = torch.sum(loss, dim=[1, 2])  # Shape: (batch_size,)
            if self.reduction == "mean":
                loss = loss / (n_ems * H * W)  # Average over all elements
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean() if self.mode == "default" else loss.mean()  # Already averaged for cnn
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError("Reduction must be 'sum', 'mean', or 'none'")
if __name__ == "__main__":
    input=torch.FloatTensor(np.ones((1,10,2,2)))
    target=torch.FloatTensor(np.ones((1,10,2,2)))
    sad=SADLoss(mode='cnn')
    print(sad(input,target))