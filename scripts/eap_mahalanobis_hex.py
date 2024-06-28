import os
from pathlib import Path
from functools import partial
from datetime import datetime as dt

from cupbearer import tasks, utils, scripts
from cupbearer.detectors import ActivationCache
from cupbearer.tasks.tiny_natural_mechanisms import get_effect_tokens

from elk_experiments.eap_detector import EAPMahalanobisDetector
from  elk_experiments.tiny_natural_mechanisms_utils import get_task_subset

from hydra.core.config_store import ConfigStore
import hydra
from omegaconf import DictConfig


config = DictConfig({
    "task_name": "hex",
    "device": "cuda",
    "model_name": "pythia-70m",
    "batch_size": 128,
    "overwrite_cache": True,
    "save_path": "output/pythia-70m-hex-filtered-eap-detector",
    "trusted_subset": None, 
    "normal_test_subset": None,
    "anomalous_test_subset": None
})

cs = ConfigStore.instance()
cs.store(name="config", node=config)

@hydra.main(version_base=None, config_name="config", config_path=None)
def train(cfg:DictConfig):
    # load task
    task = tasks.tiny_natural_mechanisms(cfg.task_name, cfg.device, cfg.model_name)
    
    # if subsets are provided, create a new task with the subsets
    if cfg.trusted_subset is not None or cfg.normal_test_subset is not None or cfg.anomalous_test_subset is not None:
        task = get_task_subset(task, cfg.trusted_subset, cfg.normal_test_subset, cfg.anomalous_test_subset)

    # set model configs for EAP
    task.model.set_use_split_qkv_input(True)
    task.model.set_use_attn_result(True)
    task.model.set_use_hook_mlp_in(True)

    # use mean probability over effect tokens as metric 
    def effect_prob_func(logits, effect_tokens):
        assert logits.ndim == 3
        # Sum over vocab and batch dim (for now we're just computing attribution values, we'll deal with per data instance later)
        probs = logits[:, -1, :].softmax(dim=-1)
        out = probs[:, effect_tokens].mean(dim=-1).mean() # mean over effect tokens, mean over batch
        # out = logits[:, -1, effect_tokens].mean()
        return out

    # get effect tokens
    effect_tokens = get_effect_tokens(cfg.task_name, task.model)
    
    # build / load activation cache
    cache_path = (Path(cfg.save_path) / "activation_cache").with_suffix(utils.SUFFIX)
    if cache_path.exists():
        eap_cache = ActivationCache.load(cache_path, device=cfg.device)
    else:
        eap_cache = ActivationCache(device=cfg.device)

    # build detector
    detector = EAPMahalanobisDetector(
        effect_prob_func=partial(effect_prob_func, effect_tokens=effect_tokens),
        upstream_nodes=["head", "mlp"],
        downstream_nodes=["head", "mlp"],
        edge_filter=lambda x: True,#not_first_layer,
        seq_len=16,
        layer_aggregation="mean", 
        cache=eap_cache
    )

    # train detector
    scripts.train_detector(
        task, detector, save_path=cfg.save_path, batch_size=cfg.batch_size, eval_batch_size=cfg.batch_size
    )
    
    # store cache
    if cache_path.exists() and cfg.overwrite_cache:
        cache_path.unlink()
    if not cache_path.exists():
        eap_cache.store(cache_path)

if __name__ == "__main__":
    train()