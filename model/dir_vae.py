import torch
import torch.nn as nn

import numpy as np

from pprint import pprint

from model.base       import BaseVAE
from model.extractors import NFINDR


class DirVAE(BaseVAE):
    r"""
        Dirichlet VAE based on "Dirichlet Variational Autoencoder"
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
                 hidden_dims: list = None,
                 encoder_activation_fn: callable = nn.LeakyReLU(),
                 encoder_batch_norm: bool = False):

        super().__init__(n_bands, n_ems)

        # beta = 1 so 1 / beta = 1
        self.one_over_beta         = torch.ones((n_ems))
        self.encoder_batch_norm    = encoder_batch_norm
        self.encoder_activation_fn = encoder_activation_fn

        if hidden_dims is None:
            self.hidden_dims = [9, 6, 3, 1]
        else:
            self.hidden_dims = hidden_dims

        self.encoder = self._build_encoder(encoder_activation_fn)
        self.decoder = self._build_decoder()


    def _build_encoder(self, encoder_activation_fn):

        encoder = []

        for i, h_dim in enumerate(self.hidden_dims):
            # dense layers
            if i == 0:
                encoder.append(nn.Linear(self.n_bands, 
                                         self.n_ems * self.hidden_dims[i]))
            else:
                encoder.append(nn.Linear(self.n_ems * self.hidden_dims[i - 1], 
                                         self.n_ems * self.hidden_dims[i]))

            # batch norms and act fn
            if i < len(self.hidden_dims) - 1:
                if self.encoder_batch_norm:
                    encoder.append(nn.BatchNorm1d(self.n_ems * self.hidden_dims[i]))
                if self.encoder_activation_fn is not None:
                    encoder.append(encoder_activation_fn)

        encoder.append(nn.Softplus())

        return nn.Sequential(*encoder)


    def _build_decoder(self):
        decoder = []
        decoder.append(nn.Linear(in_features=self.n_ems,
                       out_features=self.n_bands,
                       bias=False))
        return nn.Sequential(*decoder)


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

        v_latent = self.one_over_beta * (int4 * clamped_alphas) ** int1
        
        return v_latent


    def init_decoder(self, 
                     init_mode:str, 
                     data_spec: dict = None,
                     dataset: object = None):
        """
            Last layer is of reversed shape 
            (out_features, in_features) = (n_bands, n_ems)
        """
        
        torch.manual_seed(42)

        if init_mode == "he":
            #init_M = torch.nn.init.xavier_uniform_(self.decoder[-1].weight)
            init_M = torch.nn.init.kaiming_normal_(self.decoder[-1].weight)

        elif init_mode == "nfindr":
            nfindr = NFINDR(dataset.Y.T, data_spec["n_ems"])
            nfindr.run()
            #print(f"{nfindr.M.shape=}")
            init_M = torch.tensor(nfindr.M)

        elif init_mode == "random":
            random_indices = np.random.choice(data_spec["n_samples"], data_spec["n_ems"], replace=False)
            init_M         = torch.tensor(dataset.Y[random_indices, :].T)
            
        else:
            raise ValueError(f"Invalid init_mode, got {init_mode}")

        with torch.no_grad():  
            last_decoder_layer = self.decoder[-1]  
            last_decoder_layer.weight.copy_(init_M)
    def get_endmembers(self, 
                       layer_idx: int = -1):
        """
            Endmembers are the last layer of the decoders in Palsson AE
        """
        with torch.no_grad():
            ems_tensor = self.decoder[layer_idx].weight.data.clone()
        return ems_tensor


    def to(self, device):
        super().to(device)
        if self.one_over_beta is not None:
            self.one_over_beta = self.one_over_beta.to(device)
        return self


    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if input.ndim == 1: input = input.unsqueeze(0)

        alphas   = self.encoder(input)
        z_latent = self.process_latent(alphas)
        output   = self.decoder(z_latent)
        
        return output, z_latent, alphas


if __name__ == "__main__":
    
    from torchinfo import summary

    model = DirVAE(n_bands=10, n_ems=3, beta=1.0)
    model.debug_shape(verbose=True)

    input = torch.ones(10)
    model(input)

    summary(model, input_size=(1, 10))
