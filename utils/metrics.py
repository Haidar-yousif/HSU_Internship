import math
from utils.aligners import order_endmembers


def get_average_accuracy_on_M(pred_endmembers, gt_endmembers, criterion):
    # get endmembers and order them to be able to evaluate
    ordered_endmembers_idx = order_endmembers(pred_endmembers, gt_endmembers)

    # evaluate performance with criterion
    accuracy =  0.
    for i in range(pred_endmembers.shape[1]):
        gt_em_idx = ordered_endmembers_idx.get(i)
        if gt_em_idx is not None:
            accuracy += criterion(pred_endmembers[:, i], 
                                  gt_endmembers[:, gt_em_idx]).item()
        else:
                accuracy += math.inf
    # take the mean
    return accuracy / pred_endmembers.shape[1]


def get_accuracy_on_M(pred_endmembers, gt_endmembers, criterion):
    # get endmembers and order them to be able to evaluate
    ordered_endmembers_idx = order_endmembers(pred_endmembers, gt_endmembers)

    # evaluate performance with criterion
    accuracy =  0.
    accuracies_on_em = []

    for i in range(pred_endmembers.shape[1]):

        gt_em_idx = ordered_endmembers_idx.get(i)

        if gt_em_idx is not None:
            accuracy = criterion(pred_endmembers[:, i], 
                                 gt_endmembers[:, gt_em_idx]).item()
            accuracies_on_em.append(accuracy)
        else:
                accuracy = math.inf
                accuracies_on_em.append(accuracy)
    return ordered_endmembers_idx, accuracies_on_em

