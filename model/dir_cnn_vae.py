import torch
import torch.nn as nn

import numpy as np

from pprint import pprint


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


class DirCNNVAE(BaseVAE):
    r"""
        Dirichlet CNN VAE based on "Dirichlet Variational Autoencoder"
        by Weonyoung Joo, Wonsung Lee, Sungrae Park & Il-Chul Moon

        Theorem for reparameterization stating that
            if (X_k)_{k=1}^K \sim gamma(\alpha_k, \beta_k) i.i.d
            then if Y = (Y_1, ..., Y_K) such that \forall k
            where

        Sampling in the latent space:
            1) The encoder outputs the moments
            2) we sample
            3) we term-wise normalize by the sum

        Reparameterization trick applied on a random variable
        v \sim multigamma(\alpha, \beta \mathbb{1}_K) based
        thanks to an approximation of the inverse-CDF.
    """
    def __init__(self,
                 n_bands: int,
                 n_ems: int,
                 beta: float,
                 patch_size: int = 12,
                 init_mode='constant',
                 init_value=0.0,
                 encoder_activation_fn: callable = nn.LeakyReLU(),
                 encoder_batch_norm: bool = False):

        super().__init__(n_bands, n_ems)

        # beta = 1 so 1 / beta = 1

        self.patch_size=patch_size
        self.one_over_beta = torch.ones((n_ems,patch_size,patch_size))
        self.encoder_batch_norm    = encoder_batch_norm
        self.encoder_activation_fn = encoder_activation_fn

        self.encoder = self._build_encoder(encoder_activation_fn)
        self.decoder = self._build_decoder()
        self._init_decoder_weights(init_mode=init_mode,init_value=init_value)
    def _build_encoder(self, encoder_activation_fn):
        layers = [
            #nn.Conv2d(self.n_bands, 48, kernel_size=3,padding=1, padding_mode="reflect", bias=False),
            # nn.Conv2d(self.n_bands, 64, kernel_size=3, padding_mode="reflect", bias=False), # patch+1
            # encoder_activation_fn,
            # nn.BatchNorm2d(64),

            nn.Conv2d(self.n_bands, 16, kernel_size=3, padding_mode="reflect", bias=False), # patch+1
            encoder_activation_fn,
            nn.BatchNorm2d(16),

            nn.Conv2d(16, self.n_ems, kernel_size=1, bias=False),
            encoder_activation_fn,
            nn.BatchNorm2d(self.n_ems),
            nn.Softplus()  # Add Softplus correctly
        ]
        return nn.Sequential(*layers)


    def _build_decoder(self):
        decoder = nn.Conv2d(
            self.n_ems,
            self.n_bands,
            kernel_size=1,
            padding=0,
            padding_mode="reflect",
            bias=False,
        )

        return decoder


    def process_latent(self, alphas: torch.Tensor, eps=1e-6) -> torch.Tensor:
        r"""
            Input
            - alpha: params of the dircihlet distrib
            z_latent \sim Dir(\alpha)
        """
        v_latent = self.reparameterize(alphas)
        sum_v = torch.sum(v_latent, dim=1, keepdim=True)
        z_latent = v_latent / (sum_v + 1e-8)

        return z_latent


    def reparameterize(self, alphas: torch.Tensor) -> torch.Tensor:
        r"""
            - u \sim U(0,1)
            - v \sim multigamma(\alpha, \beta \mathbb{1}_K)

            inverse CDF of the multigamma distribution is
            v = CDF^{-1}(u ; \alpha, \beta \mathbb{1}_K) =
                          \beta^{-1}(u * \alpha * \Gamma(\alpha))^{1/\alpha}
        """
        u = torch.rand_like(alphas)

        clamped_alphas = torch.clamp(alphas, max=30) # clamped to avoid NaNs

        int1 = 1 / torch.max(clamped_alphas, 1e-8 * torch.ones_like(clamped_alphas))
        int2 = clamped_alphas.lgamma()
        int3 = int2.exp()
        int4 = int3 * u + 1e-12 # 1e-12 to avoid NaNs
        self.one_over_beta= torch.ones_like(u)

        v_latent = self.one_over_beta * (int4 * clamped_alphas) ** int1

        return v_latent



    def _init_decoder_weights(self,init_mode='constant',init_value=0.0):

      if init_mode=='constant':
        nn.init.constant_(self.decoder.weight,init_value)
        if self.decoder.bias is not None:
          nn.init.constant_(self.decoder.bias,0)
      else:
        nn.init.kaiming_normal_(self.decoder.weight,mode='fan_out',nonlinearity='relu')
        if self.decoder.bias is not None:
          nn.init.constant_(self.decoder.bias,0)

    def get_endmembers(self,
                       layer_idx: int = -1):
        """
            Endmembers are the last layer of the decoders in Palsson AE
        """
        with torch.no_grad():
            ems_tensor =  self.decoder.weight.data.mean((2, 3)).detach()
        return ems_tensor


    def to(self, device):
        """
        Moves the model and its associated buffers to a specified device.

        Parameters:
        - device (torch.device): The device to move the model to. This can be a CPU or a GPU.

        Returns:
        - self (DirCNNVAE): The model instance after moving to the specified device.
        """

        super().to(device)
        #self.one_over_beta = self.one_over_beta.to(device)
        return self



    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if input.ndim == 1: input = input.unsqueeze(0)

        alphas   = self.encoder(input)
        z_latent = self.process_latent(alphas)
        output   = self.decoder(z_latent)

        return output, z_latent, alphas


if __name__ == "__main__":

    #from torchinfo import summary

    model = DirCNNVAE(n_bands=10, n_ems=3, beta=1.0 ,patch_size=12,init_mode='constant')
    input = torch.randn(1,10,8,8)
    output,z_latent,alphas=model(input)
    print(f"Output shape: {output.shape}")
    print(f"z_latent shape: {z_latent.shape}")
    print(f"Alphas shape: {alphas.shape}")
    print(z_latent[0,:,0,0])
    print(output[0,:,0,0])
    ems=model.get_endmembers()

