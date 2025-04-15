import torch.optim as optim

from utils.optimization.tf_rms_prop import RMSpropTF

def get_optimizer(optimizer_name):
    
    if optimizer_name == "adam":
        optimizer_fn = optim.Adam
    elif optimizer_name == "sgd":
        optimizer_fn = optim.SGD
    elif optimizer_name == "tf_rms":
        optimizer_fn = RMSpropTF
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer_fn

    