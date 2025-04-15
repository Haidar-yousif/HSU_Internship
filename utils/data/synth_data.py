"""
File to generate synthetic datasets

Usage:
    1) place yourself in project root directory
    2) run 
    ```
        python utils/data/synth_data.py --alpha 4 --n_ems 3 --snr 40 --max_purities 0.8 --n_samples 4000
    ```
For more info, run 
    ``` 
        python utils/data/synth_data.py --help 
    ```
"""

import pickle
from scipy.io import loadmat, savemat
import warnings
import os

import numpy as np
from numpy.random import dirichlet
from math import sqrt

from utils.data.util_fn import open_file


def get_synth_dataset(path_to_data):
    mat = open_file(path_to_data)
    
    # spectral library
    n_materials = mat["n_materials"]
    n_outliers  = mat["n_outliers"]
    SNR         = mat["SNR"]
    sigma       = mat["sigma_noise"]
    eps         = mat["eps_noise"]

    wavelenghts = None
    max_value   = None  
    
    n_ems   = mat["M"].shape[1]
    n_bands = mat["M"].shape[0]

    # Spectra matrix
    Y = mat["Y"].T.reshape((-1, n_bands))
    # Abundance matrix
    A = mat["A"].T
    # Endmembers matrix
    M = mat["M"]  

    n_samples   = Y.shape[0] 
    n_rows      = 1
    n_cols      = n_samples

    return n_bands, n_materials, wavelenghts, n_ems, \
             n_samples, n_outliers, SNR, sigma, \
                M, A, eps, Y, max_value, \
                 n_rows, n_cols


# ------------------------------------------------------------------------------
# Mixture of truncated Dirichlet distributions
# ------------------------------------------------------------------------------
class SyntheticDataGenerator:
    """
        Class used to generate synthetic data and save it to matlab file.
        Consistent with HSIDataset class.

        Parameters
            ----------
            n_ems: int
                dimension of the Dirichlet distribution
            n_samples: int
                number of samples to generate
            modes_probability: numpy array
                distribution for the Dirichlet modes 
                When set to None, it results in all modes in concentration_sets.
            concentration_sets: numpy array (size: n_modes by n_dim)
                coefficients of the distribution
                concentration_sets[i, :] corresponds to the coefficients of the 
                i-th Dirichlet model
                When set to None, all Dirichlet distribution are taken to be uniform
            min_purities: numpy array (size: n_dim)
                minimal abundance for each endmember (default: None)
                When set to None, the minimal abundance is uniformly set equal to 0
            max_purities: numpy array (size: n_dim)
                maximal abundance for each endmember (default: None)
                When set to None, the maximal abundance is uniformly set equal to 1
    """

    def __init__(self, 
                 n_ems: int, 
                 n_samples: int,
                 min_purities: float = 0.0, 
                 max_purities: float = 0.8,
                 concentration_sets: np.array = None,
                 n_modes: int = None, 
                 modes_probability: np.array = None,
                 random_seed_for_endmembers_selections: int = 26,
                 _SNR: float = 40, 
                 n_outliers: int = 0,
                 filepath_to_spec_lib: str = './Datasets/USGS_1995_Library/USGS_1995_Library.mat'
                 ) -> None:

        self.filepath_to_spec_lib = filepath_to_spec_lib
        self.spec_lib = loadmat(filepath_to_spec_lib)['datalib']
        
        self.n_ems     = n_ems
        self.n_samples = n_samples

        self.n_bands     = self.spec_lib.shape[0]
        self.n_materials = self.spec_lib.shape[1]

        self.min_purities = min_purities
        self.max_purities = max_purities

        if n_modes is None:
            self.n_modes = 1

        if modes_probability is None:
            self.modes_probability  = np.array([1])

        if concentration_sets is None:
            self.concentration_sets = np.ones((1, n_ems))

        else:
            self.n_modes            = n_modes
            self.modes_probability  = modes_probability 
            self.concentration_sets = concentration_sets

        self.n_outliers = n_outliers
        self.SNR        = _SNR

        self.sigma      = None
        self.eps        = None

        self.random_seed_for_endmembers_selections = random_seed_for_endmembers_selections


    def generate_measurement_noise(self,
                                   M, A):
          noiseless_Y = np.dot(M, A.T) 
          self.sigma  = np.linalg.norm(noiseless_Y) / sqrt( pow(10, self.SNR / 10) )
          self.sigma /= self.n_samples
          self.eps = np.random.normal(loc=0, 
                                      scale=self.sigma, 
                                      size=(self.n_bands, self.n_samples))
          

    def generate_outliers(self, 
                          A,
                          spread=0.1):
        """
            spread: float
                constraint violation for the outliers (default: 0.1)
        """
        for n in range(self.n_outliers):
            
            #indices = np.random.permutation( np.arange(self.n_ems) )
            outlier = 1. + np.random.uniform(0, 1) * spread

            aux = np.random.uniform(0, 1, size = self.n_ems - 1)
            aux = aux / np.sum(aux) - outlier / (self.n_ems - 1)

            A[n, 0]  = outlier
            A[n, 1:] = aux
 
        return A


    def generate_abundances_matrix(self):
        """
            Build the abundance matrix A.
            A is of shape (n_samples, n_ems)
        """
        A = np.zeros((self.n_samples, self.n_ems))

        if self.n_modes is None:
            self.n_modes = 1

        # sampling from different concentrations (modes are the index)
        if self.modes_probability is None:
            modes = np.random.choice(self.n_modes, size=self.n_samples)
        else:
            modes = np.random.choice(self.n_modes, size=self.n_samples, p=self.modes_probability)

        # generate samples by controlling the purities 
        for n, mode in enumerate(modes):
            while True:
                sample = dirichlet(self.concentration_sets[mode, :])

                if((sample >= self.min_purities).all() and
                   (sample <= self.max_purities).all()):
                    A[n, :] = sample
                    break
                
        return A
    

    def generate_endmembers_matrix(self):
        """
        Generate the endmembers matrix M 
        M is of shape (n_bands, n_ems)
        """
        # local random seed
        indexes = 1 + np.random.RandomState(
                        seed=self.random_seed_for_endmembers_selections
                      ).permutation(self.n_materials - 1)
        M       = self.spec_lib[:, indexes[:self.n_ems]]
        return M
    
    
    def generate_spectra_matrix(self,
                                M, A):
        """
            Generate the dataset Y.
            Y is of shape (n_bands, n_samples).
        """
        Y = np.matmul(M, A.T) + self.eps
        return Y
        

    def generate_synthetic_data(self):
      A = self.generate_abundances_matrix()
      M = self.generate_endmembers_matrix()

      self.generate_measurement_noise(M, A)
      self.generate_outliers(A)
      Y = self.generate_spectra_matrix(M, A)

      return M, A, Y


    def generate_and_save_to_matlab_file(self,
                                         path_to_data):
        
        M, A, Y = self.generate_synthetic_data()
 
        data = {'Y': Y,
                'M': M,
                'A': A,
                'n_bands': self.n_bands,
                'n_materials': self.n_materials,
                'SNR': self.SNR,
                'sigma_noise': self.sigma,
                'eps_noise': self.eps,
                'n_outliers': self.n_outliers,
                'min_purities': self.min_purities,
                'max_purities': self.max_purities,
                'n_modes': self.n_modes,
                'concentration_sets': self.concentration_sets,
                'modes_probability': self.modes_probability,
                }
            
        print(f"file path for data generation {path_to_data=}")
        savemat(path_to_data, data)


    def save_to_pickle_file(self,
                            M, A, Y,
                            filename="synthetic_data"):
        if not self.already_saved_data: 
            data = {'Y': Y,
                    'M': M,
                    'A': A,
                    'n_bands': self.n_bands,
                    'n_materials': self.n_materials,
                    'SNR': self.SNR,
                    'sigma_noise': self.sigma,
                    'eps_noise': self.eps,
                    'n_outliers': self.n_outliers,
                    'min_purities': self.min_purities,
                    'max_purities': self.max_purities,
                    'n_modes': self.n_modes,
                    'concentration_sets': self.concentration_sets,
                    'modes_probability': self.modes_probability,
                    }
            
            with open(filename + '.pkl', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.already_saved_data = True

        else:
            warnings.warn("Saving has already been done.")
        


def get_synthetic_datasets(dataset_name="Synthetics", 
                           target_folder="./Datasets/", 
                           data_spec=None,
                           filename=None):

    folder = target_folder + dataset_name

    if filename is None:
        filename = dataset_name.lower() + ".mat"
    else:
        filename += ".mat"

    path_to_data = os.path.join(folder, filename)

    # mkdir if data dir doesn't exist
    if not os.path.isdir(folder):
        os.makedirs(folder)
        
    if not os.path.exists(path_to_data):
        if data_spec is None:
            raise ValueError("No data specification provided to generate {} dataset.".format(dataset_name))
        elif data_spec["n_ems"] is None or data_spec["n_samples"] is None:
            raise ValueError("No number of endmembers or number of samples provided.")
        else:
            data_gen = SyntheticDataGenerator(data_spec["n_ems"], 
                                              data_spec["n_samples"], 
                                              data_spec["min_purities"], 
                                              data_spec["max_purities"], 
                                              data_spec["concentration_sets"], 
                                              data_spec["n_modes"], 
                                              data_spec["modes_probability"],
                                              data_spec["random_seed_for_endmembers_selections"], 
                                              data_spec["SNR"], 
                                              data_spec["n_outliers"]
                                              )
        data_gen.generate_and_save_to_matlab_file(path_to_data)

    else:
        print(f"File {path_to_data} already exists. Skipping.")
            
    n_bands, n_materials, wavelenghts, n_ems, \
                n_samples, n_outliers, SNR, sigma, \
                M, A, eps, Y, max_value, \
                    n_rows, n_cols = get_synth_dataset(path_to_data)

    return n_bands, n_materials, wavelenghts, n_ems, \
                n_samples, n_outliers, SNR, sigma, \
                M, A, eps, Y, max_value, \
                    n_rows, n_cols


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Generate synthetic datasets with specified parameters.")
    
    parser.add_argument("--alpha", 
                        type=float, 
                        default=4.0, 
                        help="Alpha parameter for concentration sets")
    parser.add_argument("--n_ems", 
                        type=int, 
                        default=3, 
                        help="Number of endmembers")
    parser.add_argument("--snr", 
                        type=int, 
                        default=40, 
                        help="Signal-to-noise ratio")
    parser.add_argument("--n_samples", 
                        type=int, 
                        default=4000, 
                        help="Number of samples")
    parser.add_argument("--min_purities", 
                        type=float, 
                        default=0.0, 
                        help="Minimum purity value")
    parser.add_argument("--max_purities", 
                        type=float, 
                        default=0.8, 
                        help="Maximum purity value")
    parser.add_argument("--n_modes", 
                        type=int, 
                        default=1, 
                        help="Number of modes")
    parser.add_argument("--modes_probability", 
                        nargs="+", 
                        type=float, 
                        default=[1.0], 
                        help="List of mode probabilities")
    parser.add_argument("--random_seed", 
                        type=int, 
                        default=26, 
                        help="Random seed for endmember selection")
    parser.add_argument("--n_outliers", 
                        type=int, 
                        default=0, 
                        help="Number of outliers")
    parser.add_argument("--target_folder", 
                        type=str, 
                        default="./Datasets/", 
                        help="Target folder for dataset storage")

    args = parse_arguments()

    concentration_sets = args.alpha * np.ones((1, args.n_ems))

    # Data spec
    data_spec = {
        "n_ems": args.n_ems,
        "n_samples": args.n_samples,
        "SNR": args.snr,
        "min_purities": args.min_purities,
        "max_purities": args.max_purities,
        "concentration_sets": concentration_sets,
        "n_modes": args.n_modes,
        "modes_probability": args.modes_probability,
        "random_seed_for_endmembers_selections": args.random_seed,
        "n_outliers": args.n_outliers
    }

    filename_ = f"Synth_alpha_{args.alpha}_snr_{args.snr}_n_ems_{args.n_ems}"

    get_synthetic_datasets(
        dataset_name  = filename_,
        target_folder = args.target_folder,
        data_spec     = data_spec,
        filename      = filename_.lower()
    )
                