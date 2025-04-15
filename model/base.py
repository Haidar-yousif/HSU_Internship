import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class BaseVAE(nn.Module, ABC):
    """
        Skeleton for VAE-based architectures. 
    """
    def __init__(self, 
                 n_bands: int,
                 n_ems:   int):
        super().__init__()

        self.n_bands = n_bands
        self.n_ems   = n_ems

    @abstractmethod
    def _build_encoder(self) -> nn.Module:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _build_decoder(self) -> nn.Module:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def reparameterize(self, distribution_moment: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def process_latent(self, input: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError
