from tqdm import tqdm
from pprint import pprint
import math

import torch
import torch.nn as nn


class UnsupervisedTrainer:
    """
        -> Unsupervised training only
        -> Work for DirVAE
    """
    def __init__(self,  
                 device: torch.Tensor = None):
        if device is None: 
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def send_model_to_device(self, model):
        if next(model.parameters()).device != self.device:
            model = model.to(self.device)
        print("The model will be running on", 
              next(model.parameters()).device, "device.\n")

    def train(self,
              n_epochs,
              model,
              dataloader,
              optimizer,
              rec_loss_fn,
              reg_loss_fn,
              reg_factor,
              constraint,
              debug: bool,
              verbose: bool):
        """
            Training pipeline.    
            - debug: used to display numerotation in simplex fig     
        """
        self.send_model_to_device(model)
        model.train()
                           
        epoch_level_results = {
            "reconstruction_loss": [],
            "regularization_loss": [],
            "total_loss": [],
        }

        for epoch in tqdm(range(1, n_epochs + 1), leave=True):

            epoch_avg_rec_err   = 0.
            epoch_avg_reg_err   = 0.
            epoch_avg_total_err = 0.

            for batch_idx, target in enumerate(dataloader, 0):
                
                # target shape (batch size, n_bands)
                target    = target.to(self.device)
                batch_len = target.shape[0]

                optimizer.zero_grad()

                pred, z_latent, alphas = model(target)

                # losses
                batch_avg_reg_err   = reg_factor * reg_loss_fn(alphas) / batch_len
                batch_avg_rec_err   = rec_loss_fn(pred, target) / batch_len
                batch_avg_total_err = batch_avg_rec_err + batch_avg_reg_err

                # backprop
                batch_avg_total_err.backward()
                optimizer.step()

                # constraint on model's weights
                if constraint is not None:
                    constraint.apply()

                # update of losses / epoch
                epoch_avg_rec_err   += batch_avg_rec_err.item()
                epoch_avg_reg_err   += batch_avg_reg_err.item()
                epoch_avg_total_err += batch_avg_total_err.item()

            nb_batch = len(dataloader)
            epoch_avg_rec_err   /= nb_batch
            epoch_avg_reg_err   /= nb_batch
            epoch_avg_total_err /= nb_batch

            epoch_level_results["reconstruction_loss"].append(epoch_avg_rec_err)
            epoch_level_results["regularization_loss"].append(epoch_avg_reg_err)
            epoch_level_results["total_loss"].append(epoch_avg_total_err)

        return model, epoch_level_results


if __name__ == "__main__":
    # unit tests
    test = UnsupervisedTrainer()
