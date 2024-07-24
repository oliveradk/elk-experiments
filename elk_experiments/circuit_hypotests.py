from typing import Callable
from scipy.stats import binom, beta

import torch 
import numpy as np

from auto_circuit.data import PromptDataLoader, PromptPairBatch
from auto_circuit.types import CircuitOutputs, PatchType, AblationType
from auto_circuit.utils.patchable_model import PatchableModel

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

def bin_search_smallest_faithful(
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

        circuit_out = run_circuits(
            model=model, 
            dataloader=dataloader,
            test_edge_counts=[edge_count],
            prune_scores=attribution_scores,
            patch_type=PatchType.TREE_PATCH,
            ablation_type=ablation_type,
            reverse_clean_corrupt=False,
        )
        circuit_out = dict(circuit_out[edge_count])
        # model out
        model_out: CircuitOutputs = {}
        for batch in dataloader:
            model_out[batch.key] = model(batch.clean)[model.out_slice]
        # run statitiscal test 
        num_ablated_C_gt_M, n = compute_num_C_gt_M(circuit_out, model_out, dataloader, score_func)
        not_equiv, p_value = run_non_equiv_test(num_ablated_C_gt_M, n, alpha, epsilon)

        if not_equiv:
            print(f"not equiv at {edge_count}, p value : {p_value}, increase edge count")
            edge_count_interval = edge_count_interval[midpoint+1:] # more edges 
        else:
            min_equiv = edge_count
            min_equiv_p_val = p_value
            print(f"equiv at {edge_count},  p value: {p_value}, decrease edge count")
            edge_count_interval = edge_count_interval[:midpoint] # less edges
    return min_equiv, min_equiv_p_val