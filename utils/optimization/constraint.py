import torch
import torch.nn as nn


class NonNegConstraint:
    """
        Equivalent to non-neg kernel constraint in TF. 
    """
    def __init__(self, layers_list):
        self.layers_list = layers_list

    def apply(self):
        for layer in self.layers_list:
            with torch.no_grad():
                if isinstance(layer, nn.Linear):
                    if layer.weight is not None:
                        layer.weight.data.clamp_(min=0.)
                    if layer.bias is not None:
                        layer.bias.data.clamp_(min=0.)
                if isinstance(layer, nn.Parameter):
                    if layer is not None:
                        layer.data.clamp_(min=0.)
