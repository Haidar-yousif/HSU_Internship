import os
import json

import torch

from utils.visualization import plot_ems_with_gt, plot_projected_data_simplex

def create_log_folder(base_path):

    if not os.path.exists("./logs/"):
        os.makedirs(base_path)

    # list all subdir
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # find the highest number in dir names and determine the next one
    numeric_dirs = []
    for subdir in subdirs:
        if subdir.isdigit(): 
            numeric_dirs.append(int(subdir))

    next_k = max(numeric_dirs) + 1 if numeric_dirs else 0

    new_dir = os.path.join(base_path, str(next_k))
    os.makedirs(new_dir)

    return new_dir


def log_metrics(avg_sad, sad_by_ems, avg_mse, mse_by_ems, path_to_log_dir):

    metric_dict = {
        'avg_sad': avg_sad,
        'sad_by_ems': sad_by_ems,
        'avg_mse': avg_mse,
        'mse_by_ems': mse_by_ems
    }

    os.makedirs(path_to_log_dir, exist_ok=True)
    file_path = os.path.join(path_to_log_dir, "metrics.json")
    with open(file_path, "w") as file:
        json.dump(metric_dict, file, indent=4)

    return file_path

def log_figure(Y,
               pred_M,
               gt_M,
               init_M,
               path_to_log_dir,
               debug=True,
               show=True,
               verbose=False,
               save=True,
               ):
    """
        Be careful not to take the normalized version of Y.
    
    """
    # save predicted endmembers
    plot_ems_with_gt(pred_M,
                     torch.from_numpy(gt_M),
                     None,
                     save     = True,
                     filepath = os.path.join(path_to_log_dir, "predictions.pdf")
                     )

    # save simplex PCA projection in 2D
    plot_projected_data_simplex(torch.from_numpy(Y), 
                                pred_M, 
                                torch.from_numpy(gt_M), 
                                init_M, 
                                debug = True,
                                show = True,
                                save = True,
                                filepath = os.path.join(path_to_log_dir, "projected_simplex.pdf"))



