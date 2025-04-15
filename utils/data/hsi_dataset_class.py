import numpy as np
import pandas as pd
import warnings

import torch
from torch.utils.data import Dataset


DATASET_CONFIG = {
    "Samson": {
        "n_ems": 3,
        "n_pixels": 9025,
        "n_bands": 156,
    },
    "Urban4": {
        "n_ems": 4,
        "n_pixels": 94249,
        "n_bands": 162,
    },
    "Urban5": {
        "n_ems": 5,
        "n_pixels": 94249,
        "n_bands": 162,
    },
    "Urban6": {
        "n_ems": 6,
        "n_pixels": 94249,
        "n_bands": 162,
    },
}

class HSIDataset(Dataset):
    """
        Params
        ------
            Y
                Must be of shape (n_samples, n_bands)  or (n_rows, n_cols, n_bands) 
            M
                Must be of shape (n_bands, n_ems)
            A
                Must be of shape (n_samples, n_ems)
            n_cols
                Original image height
            n_rows
                Original image width
            n_sources
                Nb of endmembers
            n_bands
                Nb of frequency band
            wavelengths
                Selected wavelength
            n_samples
                Nb pixels
    """
    def __init__(self, 
                 wavelenghts, 
                 Y, M, A,
                 n_sources, n_bands, n_samples, 
                 n_rows, n_cols, 
                 dataset_name, 
                 data_type="float32",
                 random_seed=1234):

        self.wavelengths = wavelenghts
        self.n_rows      = n_rows
        self.n_cols      = n_cols
        self.n_sources   = n_sources
        self.n_bands     = n_bands
        self.n_samples   = n_samples
        self.data_type   = data_type
        self.random_seed = random_seed

        if dataset_name in DATASET_CONFIG.keys():
            config = DATASET_CONFIG[dataset_name]
            self.check_matrices_shape(Y.reshape((config["n_pixels"], config["n_bands"])),
                                      M,
                                      A,
                                      dataset_name)
        
        self.M = M
        self.A = A
        self.Y = self.select_random_Y(Y.reshape(-1, self.n_bands)) 
        # Normalize data
        self.Y /= self.Y.max()

    def __repr__(self):
        return f'HSIDataset(n_samples={self.n_samples}, \
                            n_bands={self.n_bands}, \
                            n_sources={self.n_sources}, \
                            data_type={self.data_type}'
    
    def select_random_Y(self, Y):
        """
            Randomly selects a set of pixels from the dataset.
        """
        torch.manual_seed(self.random_seed)
        pixel_idx = torch.randint(0, Y.shape[0], (self.n_samples,))
        return Y[pixel_idx, :]

    def check_matrices_shape(self, Y, M, A, dataset_name):
        """
            Matrices must be in shape:
                - Y: (n_pixels, n_bands)
                - M: (n_bands, n_ems) (doesn't change)
                - A: (n_ems, n_pixels)
        """
        if dataset_name in DATASET_CONFIG.keys():
            config   = DATASET_CONFIG[dataset_name]
            n_pixels = config["n_pixels"]
            n_bands  = config["n_bands"]
            n_ems    = config["n_ems"]

            # shape check
            assert Y.shape[0] <= n_pixels and Y.shape[0] > 0, f"Y must be of shape (n_samples <= {n_pixels}, {n_bands}), got {Y.shape}"
            assert Y.shape[1] == n_bands, f"Y must be of shape (n_samples <= {n_pixels}, {n_bands}), got {Y.shape}"
            
            assert M.shape == (n_bands, n_ems), f"M must be of shape ({n_bands}, {n_ems}), got {M.shape}"
            
            assert A.shape[0] == n_ems, f"A must be of shape ({n_ems}, n_samples <= {n_pixels}), got {A.shape}"
            assert A.shape[1] <= n_pixels and A.shape[1], f"A must be of shape ({n_ems}, n_samples <= {n_pixels}), got {A.shape}"

    def array(self):
        return self.Y.reshape((self.n_rows, self.n_cols, self.n_bands))
    
    def get_bands(self, bands):
        """
            Return the channel of the HSI corresponding to the specified band
        """
        return self.Y[:, bands]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        data = self.Y[index, :]
        data = torch.tensor(data, dtype=torch.float32) if self.data_type == "float32" else torch.tensor(data, dtype=torch.float64)
        return data 


if __name__ == "__main__":

    samson_config = DATASET_CONFIG["Samson"]
    n_bands  = samson_config["n_bands"]
    n_ems    = samson_config["n_ems"]
    n_samples = samson_config["n_pixels"]

    Y = torch.randn(n_samples, n_bands)
    M = torch.randn(n_bands, n_ems)
    A = torch.randn(n_ems, n_samples)

    dataset = HSIDataset(None, Y, M, A, n_ems, n_bands, n_samples, None, None, "Samson")
    dataset.check_matrices_shape(Y, M, A, "Samson")

