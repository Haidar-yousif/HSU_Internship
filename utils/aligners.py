import math
import torch
import numpy as np

from utils.loss.losses import SADLoss


def order_endmembers(pred_endmembers: torch.Tensor, 
                     gt_endmembers: torch.Tensor):
    """
        The goal of this function is to "order" the predicted
        endmembers by  assigning the closest gt endmembers in the 
        meaning of the SAD to the predicted endmembers.

        Keys   are the predicted endmembers
        Values are the gt        endmembers

        pred_endmembers: tensor of shape (n_bands, n_ems)
        gt_endmembers:   tensor of shape (n_bands, n_ems)
    """
    # for debug
    title = ""

    n_ems   = pred_endmembers.shape[1]
    sad     = SADLoss()
    
    # keep track of the SAD score of each predicted ems
    # with gt ones, rows correspond to predicted and 
    # columns to gt 
    sad_mat = torch.ones((n_ems, n_ems))
    index_dict = dict()

    # predicted endmember
    for i in range(n_ems):
        # gt endmember
        for j in range(n_ems):
            sad_mat[i, j] = sad(pred_endmembers[:, i], 
                                gt_endmembers[:, j])
    
    # fill a dict with predicted endmembers as keys
    # and their corresponding gt endmember as values
    nb_of_proceeded_mins = 0
    
    # continue until all predicted endmembers have been 
    # assigned a gt endmember
    while nb_of_proceeded_mins < n_ems:
        
        minimum = sad_mat.min()
        
        # return tuple( tensor(rows), tensor(columns) ) of minimum/minima
        index_arr = torch.where(sad_mat == minimum)

        # if a NaN in sad_mat raise an exception
        if math.isnan(minimum) or np.isnan(sad_mat).any():
            print("NaN value in sad_mat!")
            title += "SAD_mat contains NaN value. / "

        if minimum is None:
            print("None in sad_mat!")
            title += " None in sad_mat /"

        if len(index_arr) > 2:
            print("More than one minimum in sad_mat!")
            title += "SAD_mat contains NaN value. / "
    
        # and then stop 
        if len(index_arr) < 2 or math.isinf(minimum):
            print("break!")
            print(f"{index_arr=}")
            title += "No minimum in sad_mat / "
            break

        # take the indices of the first minimum in sad_mat
        if index_arr[0].size(dim=0) > 0 and index_arr[1].size(dim=0) > 0:
            index = (index_arr[0][0], index_arr[1][0])
        else:
            print("index_arr is empty or doesn't have enough elements.")
            print(f"{index_arr=}")
            print(f"{sad_mat=})")
            print(f"{pred_endmembers=}")
            #print(f"{gt_endmembers=}")
            raise ValueError

        # index_dict's keys are the pred ems
        # and values are the corresponding gt ems

        # if predicted endmember already has an assigned gt 
        # endmember, put infinity in sad_mat, and skip this pair
        if index[0] in index_dict.keys():
            sad_mat[index[0], index[1]] = math.inf
            print("pred endmember already assigned in dict!")
            title += "Pred endmember already assigned in dict! / "
        
        # if gt endmember already has an assigned predicted
        # endmembers put infinity in sad_mat, and skip this pair
        elif index[1] in index_dict.values():
            sad_mat[index[0], index[1]] = math.inf
            print("gt endmember already assigned in dict!")
            title += "Gt endmember already assigned in dict! / "

        # otherwise, fill the dict and put infinity in sad_mat
        # as a new min have been prosseced add 1 to counter
        else:
            index_dict[index[0].item()] = index[1].item()
            sad_mat[index[0], index[1]] = math.inf
            # withdraw unsuitable candidates for min
            sad_mat[index[0], :] = math.inf
            sad_mat[:, index[1]] = math.inf
            nb_of_proceeded_mins += 1

    # assigned the remaining gt em to the remaining pred em 
    # if necessary
    keys = index_dict.keys()
    if len(list(keys)) < n_ems:
        all_indices = set(range(n_ems))
        values = index_dict.values()
        missing_key   = all_indices - keys
        missing_value = all_indices - set(values)

        index_dict[int(next(iter(missing_key)))] = int(next(iter(missing_value)))
    
    # for debug
    if title != "": print(title)

    return index_dict




if __name__ == '__main__':
    
    pred_endmembers = torch.tensor([[2, 20, 3], 
                                    [1, 1, 1]], 
                                    dtype=torch.float32).unsqueeze(0)
    gt_endmembers   = torch.tensor([[1, 1, 1], 
                                    [2, 20, 3]], 
                                    dtype=torch.float32).unsqueeze(0)

    res = order_endmembers(pred_endmembers, gt_endmembers)
    print(f"{res=}")