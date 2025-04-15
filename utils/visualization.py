import torch

import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils.aligners import order_endmembers


def plot_reconstruction(pred_Y: torch.Tensor,
                        gt_Y: torch.Tensor):
    """
        Plot reconstructed observations
    """
    fig = plt.figure()

    np_pred_Y = pred_Y.squeeze().cpu().numpy()
    np_gt_Y   = gt_Y.squeeze().cpu().numpy()

    plt.plot(np_pred_Y / np.amax(np_pred_Y).item(), 
             label="pred_Y", 
             color="blue")

    plt.plot(np_gt_Y / np.amax(np_gt_Y).item(), 
             label="gt_Y", 
             color="red",
             linestyle="-")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.close()

    return fig


def plot_ems_with_gt(pred_M: torch.Tensor,
                     gt_M:   torch.Tensor,
                     init_M: torch.Tensor = None,
                     save = False,
                     filepath=None,
                     ):
    """
        Function to visualize endmembers spectra
        
        Args:
            pred_M: predicted endmembers
                    shape (n_bands, n_ems) 
            gt_M: ground truth endmembers
                    shape (n_bands, n_ems)
            init_M: initial endmembers, optional
                    shape (n_bands, n_ems)
    """
    n_ems = pred_M.shape[1]
  
    # assign predicted endmembers to the corresponding gt endmembers
    order_endmembers_idx = order_endmembers(pred_M, gt_M)

    fig, axes = plt.subplots(1, n_ems, figsize=(15,4))

    for i in range(n_ems):

        if init_M is not None:
            # plot init endmember
            init_em = init_M[:, i].squeeze().cpu().numpy()
            axes[i].plot(init_em / np.amax(init_em).item(), 
                        label="init_M", 
                        linestyle="dashed",
                        color="gold",
                        alpha=0.7)

        # plot gt endmember
        gt_ems_idx = order_endmembers_idx.get(i)
        gt_em = gt_M[:, gt_ems_idx].squeeze().cpu().numpy()
        axes[i].plot(gt_em / np.amax(gt_em).item(), 
                     label="gt_M", 
                     linestyle="dotted",
                     color="brown",
                     alpha=0.7)

        # plot pred endmember
        em_i = pred_M[:, i].squeeze().detach().cpu().numpy()
        axes[i].plot(em_i / np.amax(em_i).item(), 
                     label="pred_M", 
                     color="orange",
                     alpha=1.)

        axes[i].set_title(f"End members for gt ems n°{gt_ems_idx}")

        if i == 0:
            axes[i].legend()

    if save:
        #plt.tight_layout()
        plt.savefig(filepath, format="pdf", bbox_inches="tight")
    plt.close()

    return fig


def plot_projected_data_simplex(Y: torch.Tensor, 
                                pred_M: torch.Tensor = None, 
                                gt_M:   torch.Tensor = None, 
                                init_M: torch.Tensor = None, 
                                debug:    bool = False,
                                show:     bool = False, 
                                save:     bool = False,
                                filepath: str = None):
    """
        Project in 2d and display the simplex data points 
        along with the true endmembers (dots), the endmembers 
        selected at initialization (stars) and the estimated 
        endmembers (squares).
        
        Note
        ----
        This visualization is especially appropriate when the number of 
        endmembers is equal to 3. In this case, the data points are projected 
        into a subspace of dimension 2 using a Principal Component Analysis 
        algorithm.
        When one or many values is/are set to None, this/these 
        points are not displayed .
        
        Parameters
        ----------
        Y: shape (n_samples, n_bands)
            observations spectra matrix
        pred_M: shape (n_bands, n_ems)
            predicted endmembers spectra
        init_M: shape (n_bands, n_ems)
            initialized endmembers spectra
        gt_M:   numpy array of shape (n_bands, n_ems)
            ground truth endmembers spectra
        show: bool
            when False figure is not shown only returned
    """
    
    assert Y.shape[0] > Y.shape[1],           f"Please transpose Y matrix, actual shape is {Y.shape}"
    if pred_M is not None: assert pred_M.shape[0] > pred_M.shape[1], f"Please transpose pred_M matrix, actual shape is {pred_M.shape}."
    if gt_M   is not None: assert gt_M.shape[0]   > gt_M.shape[1]  , f"Please transpose gt_M matrix, actual shape is {gt_M.shape}."
    if init_M is not None: assert init_M.shape[0] > init_M.shape[1], f"Please transpose init_M matrix, actual shape is {init_M.shape}."

    # convert tensors to numpies
    n_ems = gt_M.shape[1]
    pca   = PCA(n_components = 2)
    
    Y     = Y.cpu().numpy()
    Y_max = Y.max()
    Y     = Y / Y_max

    # PCA projection
    Y_proj = pca.fit_transform(Y)

    fig = plt.figure()
    
    plt.scatter(Y_proj[:, 0], Y_proj[:, 1], 
                color='lightskyblue', 
                marker="o", 
                alpha=0.25, 
                label="Y")

    if init_M is not None:

        init_M      = init_M.cpu().numpy()
        init_M_proj = pca.transform(init_M.T)

        plt.scatter(init_M_proj[:, 0], init_M_proj[:, 1], 
                    color='gold', 
                    marker="^", 
                    label="init_M")
        if debug:
            for i in range(init_M_proj.shape[0]):
                plt.annotate("em " + str(i), (init_M_proj[i, 0], init_M_proj[i, 1]))
    
    if gt_M is not None:

        gt_M = gt_M.cpu().numpy()
        gt_M = gt_M / Y_max

        gt_M_proj = pca.transform(gt_M.T)
        plt.scatter(gt_M_proj[:, 0], gt_M_proj[:, 1], 
                    marker=(5, 1), 
                    color='brown', 
                    label="gt_M")
    
    if pred_M is not None:

        pred_M = pred_M.cpu().numpy()

        pred_M_proj = pca.transform(pred_M.T)
        plt.scatter(pred_M_proj[:, 0], pred_M_proj[:, 1], 
                    marker="s", 
                    color='orange', 
                    label="pred_M")
        if debug:
            # add the text "em i" on scatter plot
            for i in range(pred_M_proj.shape[0]):
                plt.annotate("em " + str(i), (pred_M_proj[i, 0], pred_M_proj[i, 1]))

    plt.grid()
    plt.legend()
    

    if save:
        plt.savefig(filepath, format="pdf", bbox_inches="tight")

    if show:
        plt.show()
    
    plt.close()

    return fig
