import os
import json
from pathlib import Path
from matplotlib import pyplot as plt

import torch

from transformer_lens import HookedTransformer

from cupbearer import tasks, detectors, scripts
from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.utils import SUFFIX

from eap.eap_graph import EAPGraph 
from eap.eap_wrapper import EAP

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
    Convert a path relative to the repository root to an absolute path.

    Args:
        path: A path relative to the repository root.

    Returns:
        The absolute path.
    """
    repo_abs_path = Path(__file__).parent.parent.absolute()
    return repo_abs_path / path

OUTPUT_DIR = repo_path_to_abs_path("output")


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