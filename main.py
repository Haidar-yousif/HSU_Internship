import os
import sys
import argparse
import warnings
from pathlib import Path
from pprint import pprint

parser = argparse.ArgumentParser()

_ = parser.add_argument(
    "--dataset", 
    type=str, 
    required=True, 
    default="Samson",
    help="Available datasets: \
          Samson, Urban, if synthetic will generate or use already generated ones.")
_ = parser.add_argument(
    "--target_folder", 
    type=str, 
    required=False, 
    default="./Datasets/",
    help="Folder containing the datasets.")
_ = parser.add_argument( 
    "--dataset_filename",
    type=str, 
    required=False, 
    default=None,
    help="Filename of the dataset in case of synthetic data. If not specified, keep the same as folder in lower case.")

# arguments on distribution
_ = parser.add_argument("--n_ems", 
                        type=int, 
                        required=True, 
                        default=None)
_ = parser.add_argument("--n_samples", 
                        type=int, 
                        required=False, 
                        default=4000) 
_ = parser.add_argument("--concentration_sets", 
                        type=float, 
                        required=True)
_ = parser.add_argument("--modes_probability", 
                        nargs='+', 
                        type=float,
                        help='', 
                        required=False,
                        default=[1.])
_ = parser.add_argument("--n_distr_points", 
                        type=int,   
                        required=False, 
                        default=1000, 
                        help="Number of points in the distribution.")
_ = parser.add_argument("--max_purities",   
                        type=float, 
                        required=False, 
                        default=1., 
                        help="Max purities of synthetic spectra generated for synthetic dataset.")
_ = parser.add_argument("--min_purities",   
                        type=float, 
                        required=False, 
                        default=0.,
                        help="Min purities of synthetic spectra generated for synthetic dataset.")
_ = parser.add_argument("--n_modes",        
                        type=int,   
                        required=False, 
                        default=1,
                        help="Number of modes in the Dirichlet distribution of the Synthetic \
                            generated dataset")
_ = parser.add_argument("--n_outliers",
                        type=int,   
                        required=False, 
                        default=0,
                        help="Number of outliers in the Synthetic generated dataset")
_ = parser.add_argument("--snr",            
                        type=float, 
                        required=False, 
                        default=40.,
                        help="Signal-to-noise ratio of synthetic dataset.")
_ = parser.add_argument("--data_sampling_random_seed", 
                        type=int,   
                        required=False, 
                        default=1234, 
                        help="Random seed in dataset sampling")
_ = parser.add_argument("--random_seed_for_endmembers_selections", 
                        type=int,   
                        required=False, 
                        default=26, 
                        help="Random seed used in the generation of a synthetic dataset \
                            whenever it is generated, otherwise param not used.")

# model spec
_ = parser.add_argument("--M_init",             
                        type=str, 
                        required=False, 
                        default="he",
                        help="Initialization of the model. Works for NMF model and AE model.\
                            Available options: \
                            he, nfindr and random")

# losses spec
_ = parser.add_argument("--rec_loss",    
                        type=str,   
                        required=False, 
                        default="mse",
                        help="Reconstruction loss for training. \
                            Available losses: MSE, custom_SAD, SAD_from_palsson")
_ = parser.add_argument("--reg_loss",    
                        type=str,   
                        required=False, 
                        default="gamma_div",
                        help="Regularization loss for training. \
                            Available losses: KLDiv, SinkhornDivergenceLoss")
_ = parser.add_argument("--reg_factor",      
                        type=float, 
                        required=True, 
                        default=1e-1,
                        help="Regularization factor for regularization loss on abundance.")

# training spec
_ = parser.add_argument("--n_epochs",   type=int,   required=False, default=10)
_ = parser.add_argument("--batch_size", type=int,   required=False, default=200)
_ = parser.add_argument("--optimizer",  
                        type=str,   
                        required=False, 
                        default="tf_rms",
                        help="Optimizer for training. \
                        Available optimizers: custom_rms, adam and sgd.")
_ = parser.add_argument("--lr",             type=float, required=False, default=1e-3)
_ = parser.add_argument("--momentum",       type=float, required=False, default=0.99)



# %%
import torch
import torch.nn as nn
from   torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

# %%
# customs 
from utils.data.hsi_dataset_class import HSIDataset
from utils.data.real_data         import get_dataset
from utils.data.synth_data        import get_synthetic_datasets

from model.dir_vae import DirVAE
from model.dir_cnn_vae import DirCNNVAE
from utils.optimization.constraint  import NonNegConstraint
from utils.optimization.optim_utils import get_optimizer

from utils.loss.losses_utils import (
    get_reconstruction_loss_fn,
    get_regularization_loss_fn
)

from trainer import UnsupervisedTrainer

from utils.metrics import get_average_accuracy_on_M, get_accuracy_on_M
from utils.logs_fn import create_log_folder, log_metrics, log_figure

cliargs = parser.parse_args()

print("------------------------------------------------------------")
pprint(f"RAW {cliargs=}")
print("------------------------------------------------------------")



def main():

    data_spec = dict()

    data_spec["dataset_name"]     = cliargs.dataset
    data_spec["target_folder"]    = cliargs.target_folder
    data_spec["dataset_filename"] = cliargs.dataset_filename
    data_spec["batch_size"]       = cliargs.batch_size

    data_spec["n_ems"]              = cliargs.n_ems
    data_spec["n_samples"]          = cliargs.n_samples
    data_spec["min_purities"]       = cliargs.min_purities
    data_spec["max_purities"]       = cliargs.max_purities
    data_spec["n_modes"]            = cliargs.n_modes
    data_spec["n_outliers"]         = cliargs.n_outliers
    data_spec["modes_probability"]  = np.array(cliargs.modes_probability)
    data_spec["SNR"]                = cliargs.snr

    data_spec["data_sampling_random_seed"]             = cliargs.data_sampling_random_seed
    data_spec["random_seed_for_endmembers_selections"] = cliargs.random_seed_for_endmembers_selections
    

    hyperparams = dict()

    hyperparams["n_epochs"]          = cliargs.n_epochs
    hyperparams["lr"]                = cliargs.lr
    hyperparams["momentum"]          = cliargs.momentum
    hyperparams["optimizer"]         = cliargs.optimizer
    hyperparams["rec_loss"]          = cliargs.rec_loss
    hyperparams["reg_loss"]          = cliargs.reg_loss
    hyperparams["reg_factor"]        = cliargs.reg_factor
    hyperparams["M_init"]            = cliargs.M_init
    hyperparams["device"]            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams["optimizer_epsilon"] = "default"


    ############################
    #         Dataset          #
    ############################
    real_datasets = ["Samson", "Urban4", "Urban5", "Urban6", "JasperRidge", "Cuprite"]
    if data_spec["dataset_name"] in real_datasets:
        n_bands, wavelenghts, n_ems, \
                n_samples, SNR, sigma, \
                    M, A, eps, Y, max_value, \
                    n_rows, n_cols = get_dataset(dataset_name=data_spec["dataset_name"], 
                                                 target_folder=data_spec["target_folder"])

        data_spec["wavelenghts"] = wavelenghts
        data_spec["n_ems"]       = n_ems
        data_spec["n_samples"]   = n_samples if (cliargs.n_samples is None) else cliargs.n_samples 

        data_spec["concentration_sets"] = cliargs.concentration_sets * np.array(
                                             [1.0 for i in range(data_spec["n_ems"])]
                                          ).reshape(1,-1)
    
    else:
        data_spec["n_samples"]          = cliargs.n_samples
        data_spec["n_ems"]              = cliargs.n_ems
        data_spec["concentration_sets"] = cliargs.concentration_sets * np.array(
                                            [1.0 for i in range(data_spec["n_ems"])]
                                          ).reshape(1,-1)
        
        n_bands, n_materials, wavelenghts, n_ems, \
                n_samples, n_outliers, SNR, sigma, \
                M, A, eps, Y, max_value, \
                    n_rows, n_cols = get_synthetic_datasets(dataset_name  = data_spec["dataset_name"], 
                                                            target_folder = data_spec["target_folder"], 
                                                            data_spec     = data_spec,
                                                            filename      = data_spec["dataset_filename"],)
    
    data_spec["wavelenghts"] = wavelenghts
    data_spec["n_bands"]     = n_bands

    # normalization of dataset
    dataset    = HSIDataset(data_spec["wavelenghts"],
                            Y / Y.max(), M, A, 
                            data_spec["n_ems"], data_spec["n_bands"], data_spec["n_samples"], 
                            None, None, 
                            data_spec["dataset_name"],
                            data_type="float32",
                            random_seed=data_spec["data_sampling_random_seed"])

    dataloader = DataLoader(dataset, 
                            data_spec["batch_size"], 
                            shuffle=True)   
    

    ######################################
    #               Model                #
    ######################################
    hidden_dims = [3 * data_spec["n_ems"], 
                   2 * data_spec["n_ems"], 
                   1 * data_spec["n_ems"], 
                   1]
    data_spec["hidden_dims"] = str(hidden_dims)

    model = DirCNNVAE(n_bands               = data_spec["n_bands"],
                   n_ems                 = data_spec["n_ems"],
                   beta                  = 1.,
                   hidden_dims           = hidden_dims,
                   encoder_activation_fn = nn.LeakyReLU(),
                   encoder_batch_norm    = True
                   )
    constraint = NonNegConstraint([model.decoder[0]])
    model.init_decoder(cliargs.M_init, data_spec, dataset)
    constraint.apply()

    init_M = model.get_endmembers().cpu()

    ######################################
    #               Losses               #
    ######################################
    rec_loss_fn = get_reconstruction_loss_fn(hyperparams["rec_loss"]).to(hyperparams["device"]) 
    reg_loss_fn = get_regularization_loss_fn(data_spec["concentration_sets"], hyperparams["reg_loss"]).to(hyperparams["device"])


    ######################################
    #            Optimization            #
    ######################################
    optimizer_fn = get_optimizer(hyperparams["optimizer"])
    optimizer    = optimizer_fn( model.parameters(), 
                                 hyperparams["lr"] ,
                                 hyperparams["momentum"])


    ######################################
    #              Training              #
    ######################################
    trainer = UnsupervisedTrainer()
    model, results = trainer.train(hyperparams["n_epochs"],
                                   model,
                                   dataloader,
                                   optimizer,
                                   rec_loss_fn,
                                   reg_loss_fn,
                                   hyperparams["reg_factor"],
                                   constraint,
                                   False,
                                   False
                                   )
    
    # compute results
    pred_M = model.get_endmembers().cpu()

    #sad metric
    avg_sad = get_average_accuracy_on_M(pred_M, 
                                        torch.from_numpy(dataset.M), 
                                        criterion=get_reconstruction_loss_fn("sad"))
    sad_by_ems = get_accuracy_on_M(pred_M, 
                                   torch.from_numpy(dataset.M),
                                   criterion=get_reconstruction_loss_fn("sad"))
    #mse metric
    avg_mse = get_average_accuracy_on_M(pred_M, 
                                        torch.from_numpy(dataset.M), 
                                        criterion=get_reconstruction_loss_fn("mse"))
    mse_by_ems = get_accuracy_on_M(pred_M, 
                                   torch.from_numpy(dataset.M),
                                   criterion=get_reconstruction_loss_fn("mse"))

    # create log folder to store results
    path_to_log_dir = create_log_folder("./logs")
    
    log_metrics(avg_sad, sad_by_ems, avg_mse, mse_by_ems, path_to_log_dir)
    
    # plot the simplex only for synthetic data
    if data_spec["dataset_name"] != "Samson" and not "Urban" in  data_spec["dataset_name"]:
        log_figure(Y,
                   pred_M,
                   dataset.M,
                   init_M,
                   path_to_log_dir,
                   debug=True,
                   show=True,
                   verbose=False,
                   save=True,
                   )
    

if __name__ == "__main__":
    main()