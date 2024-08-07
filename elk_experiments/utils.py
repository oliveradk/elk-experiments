import os
import json
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime
from typing import Dict, Any

import torch

from transformer_lens import HookedTransformer

from cupbearer import tasks, detectors, scripts
from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.utils import SUFFIX

from eap.eap_graph import EAPGraph 
from eap.eap_wrapper import EAP

OUTPUT_DIR = Path("output")

def set_model(model: HookedTransformer, disbale_grad: bool = True):
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.eval()
    if disbale_grad:
        for param in model.parameters():
            param.requires_grad = False
    return model

def get_activation_at_last_token(
    activation: torch.Tensor, inputs: list[list[int]], name: str
):
    if activation.ndim == 3:
        # Residual stream or equivalent, shape is (batch, seq, hidden)
        return activation[:, -1, :]
    elif activation.ndim == 4 and activation.shape[-1] == activation.shape[-2]:
        # Attention, shape is (batch, num_heads, query, key)
        # TODO: this could also be Q/K/V if n_heads happens to be head_dim
        return activation[:, :, -1, :].reshape(activation.shape[0], -1)
    elif activation.ndim == 4:
        # Query/key/value, shape is (batch, seq, num_heads, hidden)
        return activation[:, -1, :, :].reshape(activation.shape[0], -1)
    else:
        raise ValueError(f"Unexpected activation shape: {activation.shape}")


def repo_path_to_abs_path(path: str) -> Path:
    """
    (from auto-circuit)
    Convert a path relative to the repository root to an absolute path.

    Args:
        path: A path relative to the repository root.

    Returns:
        The absolute path.
    """
    repo_abs_path = Path(__file__).parent.parent.absolute()
    return repo_abs_path / path

def save_cache(data_dict: Dict[Any, Any], folder_name: str, base_filename: str):
    """
    (from auto-circuit)
    Save a dictionary to a cache file.

    Args:
        data_dict: The dictionary to save.
        folder_name: The name of the folder to save the cache in.
        base_filename: The base name of the file to save the cache in. The current date
            and time will be appended to the base filename.
    """
    folder = repo_path_to_abs_path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file_path = folder / f"{base_filename}-{dt_string}.pkl"
    print(f"Saving cache to {file_path}")
    torch.save(data_dict, file_path)


def load_cache(folder_name: str, filename: str) -> Dict[Any, Any]:
    """
    (from auto-circuit)
    Load a dictionary from a cache file.

    Args:
        folder_name: The name of the folder to load the cache from.
        filename: The name of the file to load the cache from.

    Returns:
        The loaded dictionary.
    """
    folder = repo_path_to_abs_path(folder_name)
    return torch.load(folder / filename)

def save_json(data, folder_name: str, base_filename: str): 
    """
    Save data to a json file.

    Args:
        data: The data to save.
        filename: The name of the file to save the data in.
    """
    folder = repo_path_to_abs_path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{base_filename}.json"
    print(f"Saving json to {file_path}")
    with open(file_path, "w") as f:
        json.dump(data, f)

def load_json(folder_name: str, filename: str):
    """
    Load data from a json file.

    Args:
        filename: The name of the file to load the data from.

    Returns:
        The loaded data.
    """
    folder = repo_path_to_abs_path(folder_name)
    with open(folder / filename, "r") as f:
        return json.load(f)




def prod(x):
    cum_prod = 1 
    for i in x:
        cum_prod *= i
    return cum_prod


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter