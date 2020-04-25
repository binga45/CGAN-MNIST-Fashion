# Pytorch libraries
import torch
import torch.nn.functional as F
# classic libraries
import numpy as np

import errno
import os
import shutil
import sys

def _to_onehot(var, dim):
        res = torch.zeros((var.shape[0], dim), device=device)
        res[range(var.shape[0]), var] = 1.
        return res

def create_folder(folder_path):
    """Create a folder if it does not exist.
    """
    try:
        os.makedirs(folder_path)
    except OSError as _e:
        if _e.errno != errno.EEXIST:
            raise


def to_np(var):
    """Exports torch.Tensor to Numpy array.
    """
    return var.detach().cpu().numpy()

def clear_folder(folder_path):
    """Clear all contents recursively if the folder exists.
    Create the folder if it has been accidently deleted.
    """
    create_folder(folder_path)
    for the_file in os.listdir(folder_path):
        _file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(_file_path):
                os.unlink(_file_path)
            elif os.path.isdir(_file_path):
                shutil.rmtree(_file_path)
        except OSError as _e:
            print(_e)