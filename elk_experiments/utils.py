import os
import json
from pathlib import Path
from matplotlib import pyplot as plt

import torch

from transformer_lens import HookedTransformer

from cupbearer import tasks, detectors, scripts
from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.detectors.activation_based import ActivationCache
from cupbearer.utils import SUFFIX

from eap.eap_graph import EAPGraph 
from eap.eap_wrapper import EAP

def set_model(model: HookedTransformer):
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def train_detector_cache(
        detector_dir: str, 
        detector: AnomalyDetector, 
        task: tasks.Task, 
        batch_size, 
        eval_batch_size,
        cache: ActivationCache = None,
        cache_path: str = None,
        overwrite=False,
        overwrite_cache=False,
        **train_kwargs
):  
    detector_dir = Path(detector_dir)
    if os.path.exists(detector_dir) and not overwrite:
        detector.load_weights(os.path.join(detector_dir, "detector.pth"))
        out = scripts.eval_detector(task, detector, save_path=None, batch_size=eval_batch_size, pbar=True)
    else:
        out = scripts.train_detector(
            task, detector, save_path=detector_dir, batch_size=batch_size, eval_batch_size=eval_batch_size, **train_kwargs
        )
    if overwrite_cache and cache is not None and cache_path is not None:
        if os.path.exists(cache_path): # remove old cache
            os.remove(cache_path)
        cache.store(cache_path)
    return out

def learn_graph_cache(
    model,
    tokens,
    metric,
    upstream_nodes,
    downstream_nodes,
    batch_size,
    cache_path,
    verbose=False,
    overwrite=False,
):
    if cache_path.is_file() and not overwrite:
        graph = EAPGraph(
            model.cfg, 
            upstream_nodes=upstream_nodes,
            downstream_nodes=downstream_nodes,
        )
        graph.load_scores(cache_path)
    else:
        graph = EAP(
            model=model,
            clean_tokens=tokens,
            metric=metric,
            upstream_nodes=upstream_nodes,
            downstream_nodes=downstream_nodes,
            batch_size=batch_size,
            verbose=verbose,
        )
        graph.save_scores(cache_path)
    return graph

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

def prod(x):
    cum_prod = 1 
    for i in x:
        cum_prod *= i
    return cum_prod