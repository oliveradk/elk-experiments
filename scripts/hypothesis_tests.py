#!/usr/bin/env python
# coding: utf-8

# In[2]:


# set cuda visible devices
from elk_experiments.utils import is_notebook
import os
if is_notebook():
    os.environ["CUDA_VISIBLE_DEVICES"] = "4" #"1"
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"


# # Hypothesis Testing Automatically Discovered Circuits
# 
# Procedure: 
# - Compute prune scores (via attribution patching) 
# - Search over different thresholds to find the smallest circuit where the null hypotheis of Equivalence / Dominance cannot be rejected 
# - Prune edges from circuit that are not in paths to the output, or in the case of resample ablation cannot be reached from the input
# - Test whether each edge in the circuit is minimal 
# - Test whether the circuit is complete (by seeing if the null hypothesis on the independence test can be rejected)
# 
# 

# ##  Minimal Faithful Circuit According to Prune Score Ordering

# In[4]:


import os
from typing import Callable, Dict, Tuple, Union, Optional, Any, Literal, NamedTuple
from itertools import product
from copy import deepcopy
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from scipy.stats import binom, beta

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from omegaconf import OmegaConf

from transformer_lens import HookedTransformer

from auto_circuit.types import (
    CircuitOutputs, 
    BatchKey,
    PruneScores,
    PatchType, 
    AblationType,
    SrcNode, 
    DestNode, 
    Edge,
    Node
)
from auto_circuit.data import PromptPairBatch, PromptDataLoader   
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.tensor_ops import batch_answer_diffs, batch_avg_answer_val
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.custom_tqdm import tqdm

from elk_experiments.auto_circuit.auto_circuit_utils import (
    run_circuits,
    desc_prune_scores, 
    prune_scores_threshold, 
    load_tf_model
)
from elk_experiments.auto_circuit.score_funcs import GradFunc, AnswerFunc, get_score_func

from elk_experiments.auto_circuit.circuit_hypotests import (
    get_edge_idx, 
    edges_from_mask,
    equiv_test,
    sweep_search_smallest_equiv,
    plot_num_ablated_C_gt_M, 
    plot_circuit_and_model_scores,
    compute_knees, 
    plot_edge_scores_and_knees,
    minimality_test, 
    plot_p_values, 
    plot_edge_k, 
    plot_score_quantiles,
    independence_test
)

from elk_experiments.auto_circuit.node_graph import (
    NodeGraph, 
    SeqNode, 
    NodeIdx,
    SampleType,
    get_node_idx, 
    sample_paths, 
    visualize_graph, 
    edge_in_path
)
from elk_experiments.auto_circuit.tasks import TASK_DICT
from elk_experiments.utils import OUTPUT_DIR


# In[5]:


# config class
@dataclass 
class Config: 
    task: str = "Docstring Token Circuit"
    use_abs: bool = False
    ablation_type: Union[AblationType, str] = AblationType.TOKENWISE_MEAN_CORRUPT 
    grad_func: Optional[Union[GradFunc, str]] = None
    answer_func: Optional[Union[AnswerFunc, str]] = None
    ig_samples: int = 10, 
    alpha: float = 0.05
    epsilon: Optional[float] = None
    q_star: float = 0.9 
    grad_func_mask: Optional[Union[GradFunc, str]] = None
    answer_func_mask: Optional[Union[AnswerFunc, str]] = None
    clean_corrupt: Optional[Literal["clean", "corrupt"]] = None
    side: Optional[Literal["left", "right", "none"]] = None
    out_dir: Union[str, Path] = OUTPUT_DIR 
    
    def __post_init__(self):
        if isinstance(self.ablation_type, str):
            self.ablation_type = AblationType[self.ablation_type.upper()]
        if isinstance(self.grad_func, str):
            self.grad_func = GradFunc[self.grad_func.upper()]
        elif self.grad_func is None:
            self.grad_func = GradFunc.LOGPROB if not self.use_abs else GradFunc.LOGIT
        if isinstance(self.answer_func, str):
            self.answer_func = AnswerFunc[self.answer_func.upper()]
        elif self.answer_func is None:
            self.answer_func = AnswerFunc.AVG_VAL if not self.use_abs else AnswerFunc.AVG_DIFF
        if self.epsilon is None:
            self.epsilon = 0.1 if self.use_abs else 0.0
        if isinstance(self.grad_func_mask, str):
            self.grad_func_mask = GradFunc[self.grad_func_mask.upper()]
        elif self.grad_func_mask is None:
            self.grad_func_mask = self.grad_func
        if isinstance(self.answer_func_mask, str):
            self.answer_func_mask = AnswerFunc[self.answer_func_mask.upper()]
        elif self.answer_func_mask is None:
            self.answer_func_mask = self.answer_func
        if self.clean_corrupt is None:
            self.clean_corrupt = "corrupt" if self.ablation_type == AblationType.RESAMPLE else None
        if self.side is None:
            self.side = None if self.use_abs else "left" 
        elif self.side == "none":
            self.side = None
        
        if not isinstance(self.out_dir, Path):
            self.out_dir = Path(self.out_dir)


# In[6]:


# initialize config 
conf = Config()
# get config overrides if runnign from command line
if not is_notebook():
    import sys 
    conf = OmegaConf.merge(OmegaConf.structured(conf), OmegaConf.from_cli(sys.argv[1:]))

# handle directories
conf.out_dir.mkdir(exist_ok=True)
score_dir = conf.out_dir / f"{conf.task}_{conf.ablation_type.name}_{conf.grad_func.name}_{conf.answer_func.name}_{conf.ig_samples}" 
score_dir.mkdir(exist_ok=True)
exp_dir = score_dir / f"{conf.use_abs}_{conf.alpha}_{conf.epsilon}_{conf.q_star}"
exp_dir.mkdir(exist_ok=True)


# In[ ]:


# initialize task
task = TASK_DICT[conf.task]
task.init_task()


# In[ ]:


# compute edge scores
# TODO: pass full model
prune_scores = mask_gradient_prune_scores(
    model=task.model, 
    dataloader=task.train_loader,
    official_edges=None,
    grad_function=conf.grad_func_mask.value, 
    answer_function=conf.answer_func_mask.value, #answer_function,
    mask_val=None, 
    ablation_type=conf.ablation_type,
    integrated_grad_samples=10, # 10 1 for debugging
    clean_corrupt=conf.clean_corrupt,
)


# In[ ]:


model_out_train: dict[BatchKey, torch.Tensor] = {
    batch.key: task.model(batch.clean)[task.model.out_slice] 
    for batch in task.train_loader
}
model_out_test: dict[BatchKey, torch.Tensor] = {
    batch.key: task.model(batch.clean)[task.model.out_slice] 
    for batch in task.test_loader
}


# In[ ]:


equiv_results, min_equiv = sweep_search_smallest_equiv(
    model=task.model, 
    dataloader=task.train_loader,
    prune_scores=prune_scores,
    grad_function=conf.grad_func, 
    answer_function=conf.answer_func,
    ablation_type=conf.ablation_type,
    use_abs=conf.use_abs,
    side=conf.side,
    alpha=conf.alpha,
    epsilon=conf.side,
    model_out=model_out_train,
)


# In[ ]:


equiv_test_result = equiv_test(
    model=task.model, 
    dataloader=task.test_loader,
    prune_scores=prune_scores,
    grad_func=conf.grad_func, 
    answer_function=conf.answer_func,
    ablation_type=conf.ablation_type,
    edge_counts=[min_equiv],
    use_abs=conf.use_abs,
    side=conf.side,
    alpha=conf.alpha,
    epsilon=conf.epsilon,
    model_out=model_out_test
)
print("Equiv on Test: ", not equiv_test_result[min_equiv].not_equiv)


# In[ ]:


threshold = prune_scores_threshold(prune_scores, min_equiv, use_abs=conf.use_abs)
edge_mask = {k: (torch.abs(v) if conf.use_abs else v) >= threshold for k, v in prune_scores.items()}
edges = edges_from_mask(task.model.srcs, task.model.dests, edge_mask, token=task.token_circuit)


# In[ ]:


# contruct a graph from the pruned circuit, to further prune
node_circ_graph = NodeGraph(edges, token=task.token_circuit, attn_only=task.model.cfg.attn_only)
node_circ_graph.build_graph()
assert set(node_circ_graph.parents.keys()) == set([k for k in node_circ_graph.children.keys() if k.name != "Resid Start"])


# In[ ]:


valid_edges = [
    edge for edge in edges 
    if edge_in_path(
        edge, 
        node_circ_graph, 
        reach_req=conf.ablation_type == AblationType.RESAMPLE, # only works for sample b/c E[f(X)] != f(E[X])
        in_path_req=True
    )
]
min_equiv_valid_edges = len(valid_edges)
min_equiv_valid_edges


# In[ ]:


# mask out all edges not in edges to dest
valid_edge_scores = deepcopy(prune_scores)
for edge in set(edges) - set(valid_edges):
    edge_idx = get_edge_idx(edge, tokens=task.token_circuit)
    valid_edge_scores[edge.dest.module_name][edge_idx] = 0.0


# In[ ]:


# from elk_experiments.auto_circuit.circuit_hypotests import equiv_test
# recompute equivalence 
valid_edges_equiv_result = equiv_test(
    task.model, 
    task.train_loader,
    valid_edge_scores,
    conf.grad_func,
    conf.answer_func,
    conf.ablation_type,
    thresholds=[threshold],
    model_out=model_out_train,
    full_model=None,
    use_abs=conf.use_abs,
    side=conf.side,
    alpha=conf.alpha,
    epsilon=conf.epsilon,
)[len(valid_edges)]


# In[ ]:


print(torch.topk(valid_edges_equiv_result.circ_scores - equiv_results[len(edges)].circ_scores, 4))
assert not valid_edges_equiv_result.not_equiv
assert torch.allclose(valid_edges_equiv_result.circ_scores, equiv_results[len(edges)].circ_scores, atol=1e-1) # should make a function of model size


# In[ ]:


fig = draw_seq_graph(
    model=task.model,
    prune_scores=valid_edge_scores,
    score_threshold=threshold,
    show_all_seq_pos=True,
    orientation="h",
    use_abs=False,
    seq_labels=task.test_loader.seq_labels,
    file_path=exp_dir / "valid_edge_graph.png"
)


# In[ ]:


fig, ax = plot_num_ablated_C_gt_M(equiv_results, epsilon=conf.epsilon, min_equiv=min_equiv, side="left" if not conf.use_abs else None)
fig.savefig(exp_dir / "num_ablated_C_gt_M.png")


# In[ ]:


fig, ax = plot_circuit_and_model_scores(equiv_results, min_equiv)
fig.savefig(exp_dir / "circuit_model_scores.png")


# In[ ]:


# plot attribution scores 
import numpy as np
edge_scores = np.flip(desc_prune_scores(prune_scores, use_abs=conf.use_abs).detach().cpu().numpy())
if not conf.use_abs:
    edge_scores = edge_scores[edge_scores > 0]
kneedle_poly, kneedle_1d = compute_knees(edge_scores)
fig, ax = plot_edge_scores_and_knees(edge_scores, kneedle_poly, kneedle_1d, min_equiv)
fig.savefig(exp_dir / "edge_scores_knees.png")


# In[ ]:


round(len(edge_scores) - kneedle_poly.knee), round(len(edge_scores) - kneedle_1d.knee), min_equiv


# # Minimality Test

# In[ ]:


# build full grap to sample paths
node_graph = NodeGraph(task.model.edges, token=task.token_circuit, attn_only=task.model.cfg.attn_only)
node_graph.build_graph()


# In[ ]:


if valid_edges_equiv_result.not_equiv:
    edges_under_test = edges 
    edges_under_test_scores = prune_scores
else: 
    edges_under_test = valid_edges
    edges_under_test_scores = valid_edge_scores
edges_under_test_scores = {edge: edges_under_test[edge.dest.module_name][get_edge_idx(edge, tokens=task.token_circuit)] for edge in edges_under_test}
edges_under_test = sorted(edges_under_test_scores.keys(), key=lambda x: abs(edges_under_test_scores[x]), reverse=False)


# In[ ]:


# sample paths to be used for testing
n_paths = 256 
filtered_paths_walk = sample_paths(node_graph, n_paths, SampleType.WALK, edges_under_test)
mean_walk = np.mean([len(path) for path in filtered_paths_walk])
filtered_paths_uniform = sample_paths(node_graph, n_paths, SampleType.UNIFORM, edges_under_test)
mean_uniform = np.mean([len(path) for path in filtered_paths_uniform])
mean_walk, mean_uniform


# ## Run Minimality Tests

# In[ ]:


min_test_results = minimality_test(
    model=task.model, 
    dataloader=task.test_loader,
    prune_scores=edges_under_test_scores,
    edges=edges_under_test, 
    edge_count=len(edges_under_test),
    ablation_type=conf.ablation_type,
    grad_function=conf.grad_func,
    answer_function=conf.answer_func,
    filtered_paths=filtered_paths_walk,
    use_abs=conf.use_abs,
    tokens=task.token_circuit,
    alpha=conf.alpha, 
    q_star=conf.q_star
)


# In[ ]:


# minimality test on true edges
true_edge_prune_scores = {mod_name: torch.zeros_like(score_func) for mod_name, score_func in prune_scores.items()}
for edge in task.true_edges:
    true_edge_prune_scores[edge.dest.module_name][get_edge_idx(edge, tokens=task.token_circuit)] = 1.0

filtered_paths_true_edge = sample_paths(node_graph, n_paths, SampleType.WALK, task.true_edges)
min_test_true_edge_results = minimality_test(
    model=task.model, 
    dataloader=task.test_loader,
    prune_scores=true_edge_prune_scores,
    edges=task.true_edges, 
    edge_count=task.true_edge_count,
    ablation_type=conf.ablation_type,
    grad_function=conf.grad_func,
    answer_function=conf.answer_func,
    filtered_paths=filtered_paths_true_edge,
    use_abs=conf.use_abs,
    tokens=task.token_circuit,
    alpha=conf.alpha, 
    q_star=conf.q_star,
    stop_if_failed=False
)


# In[ ]:


# plot p values as scatter plot
#TODO: I think there's an error here? 
fig, ax = plot_p_values(min_test_results, edges_under_test, edges_under_test_scores)


# In[ ]:


true_edge_scores = {edge: torch.tensor(1.1) for edge in task.true_edges}
fig, ax = plot_p_values(min_test_true_edge_results, task.true_edges, true_edge_scores)


# In[ ]:


# plot frac of n 
batch_size = task.batch_size[1] if isinstance(task.batch_size, tuple) else task.batch_size
batch_count = task.batch_count[1] if isinstance(task.batch_count, tuple) else task.batch_count
plot_edge_k(min_test_results, edges_under_test, edges_under_test_scores, batch_size * batch_count, q_star=conf.q_star)


# In[ ]:


batch_size = task.batch_size[1] if isinstance(task.batch_size, tuple) else task.batch_size
batch_count = task.batch_count[1] if isinstance(task.batch_count, tuple) else task.batch_count
plot_edge_k(min_test_true_edge_results, task.true_edges, true_edge_scores, batch_size * batch_count, q_star=conf.q_star)


# In[ ]:


# plot average diff 
fit, ax = plot_score_quantiles(min_test_results, edges_under_test, edges_under_test_scores, quantile_range=[0.00, 1.00])


# In[ ]:


fit, ax = plot_score_quantiles(min_test_true_edge_results, task.true_edges, true_edge_scores, quantile_range=[0.00, 1.00])


# # Independence Test
# Test for completeness - if the circuit contains all the components required to perform the task, then the output of the complement should be independent of the original model
# 
# $H_0$: Score of complement indepedendent of score of model
# 
# Hilbert Schmdit Indepednence Criterion - non-parametric measure of independence 
# 
# - Background: (see https://jejjohnson.github.io/research_journal/appendix/similarity/hsic/)
# 
# Intuition: the trace sums along the interaction terms on each data point, which 
# we expect to be larger then other interaction terms across samples if X, and Y are 
# correlated, fewer of the perumations should be greater, our p-value will be smaller, 
# and thus we're more likely to reject the null
# 
# 
# Note: the hypothesis paper defines HCIC as  K_{x,y}K_{x,y}, but can also define it as 
# {K_x}{K_y}, b/c that that equality holds in general for Cross Covariance and Auto 
# Covariance 
# 
# The paper uses $\rho$ = median(||score(complement) - score(model)||), based on this 
# paper https://arxiv.org/pdf/1707.07269
# 
# 
# 

# In[ ]:


indep_result = independence_test(
    model=task.model, 
    dataloader=task.test_loader, 
    prune_scores=edges_under_test_scores, 
    grad_function=conf.grad_func,
    answer_function=conf.answer_func,
    threshold=threshold, 
    use_abs=conf.use_abs,
    B=1000
)
indep_result


# In[ ]:


indep_true_edge_result = independence_test(
    task.model, 
    task.test_loader, 
    true_edge_prune_scores, 
    grad_function=conf.grad_func,
    answer_function=conf.answer_func,
    threshold=0.9, 
    use_abs=True,
    alpha=conf.alpha,
    B=1000
)
indep_true_edge_result
