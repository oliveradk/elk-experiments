from typing import Callable, Dict, Tuple, Union, Optional, Any
from itertools import product
from copy import deepcopy
import random

import torch 
import numpy as np
from scipy.stats import binom, beta

from auto_circuit.data import PromptDataLoader, PromptPairBatch
from auto_circuit.types import (
    CircuitOutputs, 
    BatchKey,
    PruneScores,
    PatchType, 
    AblationType,
    SrcNode, 
    DestNode, 
    Edge
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import prune_scores_threshold

from elk_experiments.auto_circuit_utils import run_circuits



def compute_num_C_gt_M(
    circ_out: CircuitOutputs, 
    model_out: CircuitOutputs, 
    dataloader: PromptDataLoader, 
    score_func: Callable[[torch.Tensor, PromptPairBatch], torch.Tensor]
) -> tuple[int, int]:
    # compute number of samples with ablated C > M
    num_ablated_C_gt_M = 0
    n = 0
    for batch in dataloader:
        bs = batch.clean.size(0)
        circ_out_batch = circ_out[batch.key]
        model_out_batch = model_out[batch.key]
        circ_score = score_func(circ_out_batch, batch)
        model_score = score_func(model_out_batch, batch)
        num_ablated_C_gt_M += torch.sum(circ_score > model_score).item()
        n += bs
    return num_ablated_C_gt_M, n 

def run_non_equiv_test(num_ablated_C_gt_M: int, n: int, alpha: float = 0.05, epsilon: float = 0.1) -> tuple[bool, float]:
    theta = 1 / 2 + epsilon
    k = num_ablated_C_gt_M
    left_tail = binom.cdf(min(n-k, k), n, theta)
    right_tail = 1 - binom.cdf(max(n-k, k), n, theta)
    p_value = left_tail + right_tail
    return p_value < alpha, p_value 

def bernoulli_range_test(K,N,eps=0.1,a=[1,1],alpha=0.5):
    #Inputs:
    #  K: number of successes
    #  N: number of trials
    #  eps: faithfulness threshold
    #  a: beta prior coefficients on pi
    #  alpha: rejection threshold  
    #Outputs: 
    #  p(0.5-eps <= pi <= 0.5+eps | N, K, a)
    #  p(0.5-eps <= pi <= 0.5+eps | N, K, a)<1-alpha

    p_piK     = beta(N-K+a[0],K+a[1])
    p_between = p_piK.cdf(0.5+eps) - p_piK.cdf(0.5-eps)
    return(p_between<1-alpha, p_between)

def equiv_test(
    model: PatchableModel, 
    dataloader: PromptDataLoader,
    attribution_scores: torch.Tensor,
    score_func: Callable[[torch.Tensor, PromptPairBatch], torch.Tensor],
    ablation_type: AblationType,
    edge_counts: list[int],
    alpha: float = 0.05,
    epsilon: float = 0.1,
) -> Dict[int, Tuple[int, int, bool, float]]:
    
    # circuit out
    circuit_outs = dict(run_circuits(
        model=model, 
        dataloader=dataloader,
        test_edge_counts=edge_counts,
        prune_scores=attribution_scores,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
    ))
    # model out
    model_out: CircuitOutputs = {}
    for batch in dataloader:
        model_out[batch.key] = model(batch.clean)[model.out_slice]
    # run statitiscal tests for each edge count
    test_results = {}
    for edge_count, circuit_out in circuit_outs.items():
        num_ablated_C_gt_M, n = compute_num_C_gt_M(circuit_out, model_out, dataloader, score_func)
        not_equiv, p_value = run_non_equiv_test(num_ablated_C_gt_M, n, alpha, epsilon)
        test_results[edge_count] = (num_ablated_C_gt_M, n, not_equiv, p_value)
    return test_results

def bin_search_smallest_equiv(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    attribution_scores: torch.Tensor,
    score_func: Callable[[torch.Tensor, PromptPairBatch], torch.Tensor],
    ablation_type: AblationType, 
    alpha: float = 0.05,
    epsilon: float = 0.1,
):
    edge_count_interval = [i for i in range(model.n_edges + 1)]
    min_equiv = edge_count_interval[-1]
    min_equiv_p_val = 0.0
    while len(edge_count_interval) > 0:
        midpoint = len(edge_count_interval) // 2
        edge_count = edge_count_interval[midpoint]

        num_ablated_C_gt_M, n, not_equiv, p_value = equiv_test(
            model=model, 
            dataloader=dataloader,
            attribution_scores=attribution_scores,
            score_func=score_func,
            ablation_type=ablation_type,
            edge_counts=[edge_count],
            alpha=alpha,
            epsilon=epsilon,
        )[edge_count]

        if not_equiv:
            print(f"not equiv at {edge_count}, p value : {p_value}, increase edge count")
            edge_count_interval = edge_count_interval[midpoint+1:] # more edges 
        else:
            min_equiv = edge_count
            min_equiv_p_val = p_value
            print(f"equiv at {edge_count},  p value: {p_value}, decrease edge count")
            edge_count_interval = edge_count_interval[:midpoint] # less edges
    return min_equiv, min_equiv_p_val


def create_paths(srcs: set[SrcNode], dests: set[DestNode], n_layers: int) -> list[list[Edge]]:
    srcs_by_layer = {layer: {} for layer in range(1, n_layers+1)}
    for scr in srcs:
        if scr.layer < 1:
            continue
        srcs_by_layer[scr.layer][scr.head_idx] = scr

    dests_by_layer = {layer: [] for layer in range(1, n_layers+1)}
    for dest in dests:
        if dest.layer > n_layers:
            continue
        dests_by_layer[dest.layer].append(dest)
    for layer in dests_by_layer.keys():
        dests_by_layer[layer].append(None)

    paths: list[tuple[DestNode]] = list(product(*[layer_dests for layer_dests in dests_by_layer.values()]))

    start = next((n for n in srcs if n.name == "Resid Start"))
    end = next((n for n in dests if n.name == "Resid End"))

    paths_edges: list[list[Edge]] = []
    for path in paths:
        path_edges = []
        cur_src = start 
        for dest in path:
            if dest is None:
                continue
            path_edges.append(Edge(src=cur_src, dest=dest))
            cur_src = srcs_by_layer[dest.layer][dest.head_idx]
        path_edges.append(Edge(src=cur_src, dest=end))
        paths_edges.append(path_edges)
    return paths_edges

def edges_from_mask(srcs: set[SrcNode], dests: set[DestNode], mask: Dict[str, torch.Tensor]) -> list[Edge]:
    SRC_IDX_TO_NODE = {src.src_idx: src for src in srcs}
    DEST_MOD_AND_HEAD_TO_NODE = {(dest.module_name, dest.head_idx): dest for dest in dests}
    edges = []
    for mod_name, mask in mask.items():
        for idx in mask.nonzero():
            if len(idx) == 1:
                dest_node = DEST_MOD_AND_HEAD_TO_NODE[(mod_name, None)]
                src_node = SRC_IDX_TO_NODE[idx.item()]
            else:
                dest_node = DEST_MOD_AND_HEAD_TO_NODE[(mod_name, idx[0].item())]
                src_node = SRC_IDX_TO_NODE[idx[1].item()]
            edges.append(Edge(src=src_node, dest=dest_node))
    return edges

def make_complement_paths(
    srcs: set[SrcNode], 
    dests: set[DestNode], 
    n_layers: int, 
    edges: list[Edge],
) -> list[list[Edge]]:
    paths = create_paths(srcs, dests, n_layers)
    complement_paths = [
        path for path in paths if not set(path).intersection(edges)
    ]
    return complement_paths

def get_edge_idx(edge: Edge):
    if edge.dest.name == "Resid End":
        return (edge.src.src_idx,)
    return (edge.dest.head_idx, edge.src.src_idx)


def set_score(edge: Edge, scores, value, batch_idx: Optional[int] = None):
    idx = get_edge_idx(edge)
    if batch_idx is not None:
        idx = (batch_idx,) + idx
    scores[edge.dest.module_name][idx] = value
    return scores

def join_values(d: Dict[Any, Dict]):
    return {k: v for sub_d in d.values() for k, v in sub_d.items()}

def minimality_test(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    attribution_scores: torch.Tensor,
    edges: list[Edge], 
    filtered_paths: list[list[Edge]],
    edge_count: int, 
    ablation_type: AblationType, 
    score_func: Callable[[torch.Tensor, PromptPairBatch], torch.Tensor],
    circuit_out: Optional[CircuitOutputs] = None,
    threshold: Optional[float] = None,
    alpha: float = 0.05,
    q_star: float = 0.9,
) -> Dict[Edge, Tuple[bool, float, int, list[torch.Tensor], list[torch.Tensor]]]:
    if circuit_out is None:
        circuit_out = dict(run_circuits(
            model=model, 
            dataloader=dataloader,
            test_edge_counts=[edge_count],
            prune_scores=attribution_scores,
            patch_type=PatchType.TREE_PATCH,
            ablation_type=ablation_type,
            reverse_clean_corrupt=False,
        ))[edge_count]
    if threshold is None:
        threshold = prune_scores_threshold(attribution_scores, edge_count)
    test_results = {}
    for edge in edges:
        test_results[edge] = minimality_test_edge(
            model=model,
            dataloader=dataloader,
            attribution_scores=attribution_scores,
            edge=edge,
            filtered_paths=filtered_paths,
            ablation_type=ablation_type,
            threshold=threshold,
            score_func=score_func,
            circuit_out=circuit_out,
            alpha=alpha / len(edges), # bonferroni correction
            q_star=q_star,
        )
    return test_results

def minimality_test_edge(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    attribution_scores: torch.Tensor,
    edge: Edge,
    filtered_paths: list[list[Edge]],
    ablation_type: AblationType,
    threshold: float,
    score_func: Callable[[torch.Tensor, PromptPairBatch], torch.Tensor],
    circuit_out: Optional[CircuitOutputs] = None,
    alpha: float = 0.05,
    q_star: float = 0.9,
) -> Tuple[bool, float, int, list[torch.Tensor], list[torch.Tensor]]:
    
    # ablate edge and run 
    prune_scores_ablated = deepcopy(attribution_scores)
    prune_scores_ablated[edge.dest.module_name][get_edge_idx(edge)] = 0.0
    circuit_out_ablated = join_values(run_circuits(
        model=model, 
        dataloader=dataloader,
        thresholds=[threshold],
        prune_scores=prune_scores_ablated,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
    ))

    # sample random paths, inflate prune scores, and run
    sampled_paths: Dict[BatchKey, list[Edge]] = {}
    prune_scores_inflated: Dict[BatchKey, PruneScores] = {}
    for batch in dataloader:
        prune_scores_inflated[batch.key] = {
            k: score.unsqueeze(0).repeat_interleave(batch.clean.size(0), 0)
            for k, score in attribution_scores.items()
        }
        sampled_paths[batch.key] = random.choices(filtered_paths, k=batch.clean.size(0))
    for batch_key, paths in sampled_paths.items():
        for batch_idx, path in enumerate(paths):
            for edge in path:
                set_score(edge, prune_scores_inflated[batch_key], float("inf"), batch_idx=batch_idx)
    circuit_out_inflated = join_values(run_circuits(
        model=model, 
        dataloader=dataloader,
        thresholds=[threshold],
        prune_scores=prune_scores_inflated,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
    ))

    # ablate random edges in paths and run 
    prune_scores_ablated_paths = prune_scores_inflated
    for batch_key, paths in sampled_paths.items():
        for batch_idx, path in enumerate(paths):
            edge_to_ablate = random.choice(path)
            set_score(edge_to_ablate, prune_scores_ablated_paths[batch_key], 0.0, batch_idx)
    circuit_out_ablated_paths = join_values(run_circuits(
        model=model, 
        dataloader=dataloader,
        thresholds=[threshold],
        prune_scores=prune_scores_ablated_paths,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
    ))

    # compute statistics
    n = 0
    num_edge_score_gt_ref = 0
    diffs = []
    diffs_inflated = []
    for batch in dataloader:
        bs = batch.clean.size(0)
        # compute frequency diff between full circuit and ablated edge is greater than inflated circuit - ablated circuit
        circ_out_ablated = circuit_out_ablated[batch.key]
        circ_out_inflated = circuit_out_inflated[batch.key]
        circ_out_ablated_paths = circuit_out_ablated_paths[batch.key]
        circ_out = circuit_out[batch.key]
        circ_out_logit_diff = score_func(circ_out, batch)
        circ_out_ablated_logit_diff = score_func(circ_out_ablated, batch)
        circ_out_inflated_logit_diff = score_func(circ_out_inflated, batch)
        circ_out_inflated_ablated_logit_diff = score_func(circ_out_ablated_paths, batch)

        circ_diff = torch.abs(circ_out_logit_diff - circ_out_ablated_logit_diff)
        circ_inflated_diff = torch.abs(circ_out_inflated_logit_diff - circ_out_inflated_ablated_logit_diff)
        num_edge_score_gt_ref += torch.sum(circ_diff > circ_inflated_diff).item()
        # log diffs
        diffs.append(circ_diff)
        diffs_inflated.append(circ_inflated_diff)
        n += bs
    
    p_value = binom.cdf(num_edge_score_gt_ref, n, q_star)
    return alpha < p_value, p_value, num_edge_score_gt_ref, diffs, diffs_inflated
    



