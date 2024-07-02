import os
import json
from pathlib import Path
from matplotlib import pyplot as plt

from cupbearer import tasks, detectors, scripts
from cupbearer.detectors.anomaly_detector import AnomalyDetector
from cupbearer.detectors.activation_based import ActivationCache
from cupbearer.utils import SUFFIX

from eap.eap_graph import EAPGraph 
from eap.eap_wrapper import EAP


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