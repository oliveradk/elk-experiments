from typing import Callable, Dict, Tuple, Union, Optional, Any, Literal, NamedTuple
from itertools import product
from copy import deepcopy
import random
import math

import torch 
import numpy as np
from scipy.stats import binom, beta

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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
from auto_circuit.utils.custom_tqdm import tqdm

from elk_experiments.auto_circuit.auto_circuit_utils import run_circuits, load_tf_model
from elk_experiments.auto_circuit.score_funcs import GradFunc, AnswerFunc, get_score_func



def compute_num_C_gt_M(
    circ_out: CircuitOutputs, 
    model_out: CircuitOutputs, 
    dataloader: PromptDataLoader, 
    grad_function: GradFunc,
    answer_function: AnswerFunc,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    # compute number of samples with ablated C > M
    score_func = get_score_func(grad_function, answer_function)
    num_ablated_C_gt_M = 0
    n = 0
    circ_scores = []
    model_scores = []
    for batch in dataloader:
        bs = batch.clean.size(0)
        circ_out_batch = circ_out[batch.key]
        model_out_batch = model_out[batch.key]
        circ_score = score_func(circ_out_batch, batch)
        model_score = score_func(model_out_batch, batch)
        num_ablated_C_gt_M += torch.sum(circ_score > model_score).item()
        n += bs
        circ_scores.append(circ_score)
        model_scores.append(model_score)
    return num_ablated_C_gt_M, n, torch.cat(circ_scores), torch.cat(model_scores)

# ok 
def run_non_equiv_test(
    num_ablated_C_gt_M: int, 
    n: int, 
    alpha: float = 0.05, 
    epsilon: float = 0.1, 
    side: Optional[Literal["left", "right"]] = None
) -> tuple[bool, float]:
    k = num_ablated_C_gt_M
    if side == "left": # binomial test for p_success < 0.5 - epsilon
        theta = 1 / 2 - epsilon
        p_value = binom.cdf(k, n, theta)
    elif side == "right": # binomail test for p_success > 0.5 + epsilon (typically don't use)
        theta = 1 / 2 + epsilon
        p_value = 1 - binom.cdf(k, n, theta)
    else: # standard two-tailed test from the paper
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

class EquivResult(NamedTuple):
    num_ablated_C_gt_M: int
    n: int
    not_equiv: bool
    p_value: float
    circ_scores: torch.Tensor
    model_scores: torch.Tensor

def equiv_test(
    model: PatchableModel, 
    dataloader: PromptDataLoader,
    attribution_scores: torch.Tensor,
    grad_function: GradFunc,
    answer_function: AnswerFunc,
    ablation_type: AblationType,
    edge_counts: Optional[list[int]] = None,
    thresholds: Optional[list[float]] = None,
    use_abs: bool = True,
    model_out: Optional[Dict[BatchKey, torch.Tensor]] = None,
    full_model: Optional[torch.nn.Module] = None,
    side: Optional[Literal["left", "right"]] = None,
    alpha: float = 0.05,
    epsilon: float = 0.1,
) -> Dict[int, EquivResult]:

    # circuit out
    circuit_outs = dict(run_circuits(
        model=model, 
        dataloader=dataloader,
        test_edge_counts=edge_counts,
        thresholds=thresholds,
        prune_scores=attribution_scores,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
        use_abs=use_abs,
    ))
    # model out
    if model_out is None:
        model_out = {}
        ref_model = full_model if full_model is not None else model
        for batch in dataloader:
            model_out[batch.key] = ref_model(batch.clean)[model.out_slice]
    # run statitiscal tests for each edge count
    test_results = {}
    for edge_count, circuit_out in circuit_outs.items():
        num_ablated_C_gt_M, n, circ_scores, model_scores = compute_num_C_gt_M(
            circuit_out, model_out, dataloader, grad_function, answer_function
        )
        not_equiv, p_value = run_non_equiv_test(num_ablated_C_gt_M, n, alpha, epsilon, side=side)
        test_results[edge_count] = EquivResult(num_ablated_C_gt_M, n, not_equiv, p_value, circ_scores, model_scores)
    return test_results



def sweep_search_smallest_equiv(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    attribution_scores: torch.Tensor,
    grad_function: GradFunc,
    answer_function: AnswerFunc,
    ablation_type: AblationType, 
    use_abs: bool = True,
    side: Optional[Literal["left", "right"]] = None,
    alpha: float = 0.05,
    epsilon: float = 0.1,
) -> tuple[dict[int, EquivResult], int]:
    """Returns equiv test results and minimal equivalent number of edges."""
    full_results = {}
    width = 10 ** math.floor(math.log10(model.n_edges)-1)
    interval_min = 0 
    interval_max = model.n_edges
    model_out = {batch.key: model(batch.clean)[model.out_slice] for batch in dataloader}
    while width > 0:
        print(f"interval: {interval_min} - {interval_max}")
        print("width", width)
        edge_counts = [i for i in range(interval_min, interval_max, width)]
        edge_counts.append(interval_max)
        test_results = equiv_test(
            model, 
            dataloader,
            attribution_scores,
            grad_function,
            answer_function,
            ablation_type,
            edge_counts,
            model_out=model_out,
            full_model=None,
            use_abs=use_abs,
            side=side,
            alpha=alpha,
            epsilon=epsilon,
        )
        full_results.update(test_results)
        # find lowest interval where equivalence holds
        equivs = [k for k, v in test_results.items() if not v.not_equiv]
        min_equiv = min(equivs) if equivs else model.n_edges
        # round up to width or n_edges
        if min_equiv % width != 0:
            min_equiv = min(min_equiv + width - min_equiv % width, model.n_edges)
        # cases
        new_width = width // 10
        if min_equiv == model.n_edges:
            interval_max = model.n_edges
            interval_min = model.n_edges - model.n_edges % width - width
        else:
            interval_max = min_equiv
            interval_min = min_equiv - width
        width = new_width
    del model_out
    full_results = {k: full_results[k] for k in sorted(full_results.keys())}
    return full_results, interval_max
    


def bin_search_smallest_equiv(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    attribution_scores: torch.Tensor,
    grad_function: GradFunc,
    answer_function: AnswerFunc,
    ablation_type: AblationType, 
    use_abs: bool = True,
    side: Optional[Literal["left", "right"]] = None,
    alpha: float = 0.05,
    epsilon: float = 0.1,
):
    edge_count_interval = [i for i in range(model.n_edges + 1)]
    min_equiv = edge_count_interval[-1]
    min_equiv_p_val = 0.0
    model_out = {batch.key: model(batch.clean)[model.out_slice] for batch in dataloader}
    while len(edge_count_interval) > 0:
        midpoint = len(edge_count_interval) // 2
        edge_count = edge_count_interval[midpoint]

        num_ablated_C_gt_M, n, not_equiv, p_value = next(iter(equiv_test(
            model=model, 
            dataloader=dataloader,
            attribution_scores=attribution_scores,
            grad_function=grad_function,
            answer_function=answer_function,
            ablation_type=ablation_type,
            edge_counts=[edge_count],
            model_out=model_out,
            use_abs=use_abs,
            side=side,
            alpha=alpha,
            epsilon=epsilon,
        ).values()))

        if not_equiv:
            print(f"not equiv at {edge_count}, p value : {p_value}, increase edge count")
            edge_count_interval = edge_count_interval[midpoint+1:] # more edges 
        else:
            min_equiv = edge_count
            min_equiv_p_val = p_value
            print(f"equiv at {edge_count},  p value: {p_value}, decrease edge count")
            edge_count_interval = edge_count_interval[:midpoint] # less edges
    del model_out
    return min_equiv, min_equiv_p_val


def plot_equivs_bar(results: dict[int, EquivResult]):
    fig, ax = plt.subplots(figsize=(20, 1))
    not_equivs = [results[edge_count][2] for edge_count in results.keys()]
    data = not_equivs
    # Create a custom colormap
    cmap = mcolors.ListedColormap(['green', 'red'])
    norm = mcolors.BoundaryNorm([0, 0.5, 1], cmap.N)

    # Create the bar plot
    ax.bar(edge_counts, [1]*len(data), color=cmap(norm(data)), width=10)

    # adjust plot
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xlim(-0.5, max(edge_counts) - 0.5)  # Adjust x-axis limits to center the bars

    # add label
    ax.set_xlabel("edge count")

    # Create legend
    red_patch = mpatches.Patch(color='red', label='Not Equiv')
    green_patch = mpatches.Patch(color='green', label='Equiv')
    ax.legend(handles=[red_patch, green_patch], loc='upper center', bbox_to_anchor=(0.15, -0.25), ncol=2)

    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig, ax


import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, Any

def plot_num_ablated_C_gt_M(results: Dict[int, Any], min_equiv: int, epsilon: float = 0.1) -> Tuple[plt.Figure, plt.Axes]:
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be a float between 0 and 1")

    # Extract data from results
    edge_counts = list(results.keys())
    num_ablated_C_gt_Ms = [results[edge_count][0] for edge_count in edge_counts]
    ns = [results[edge_count][1] for edge_count in edge_counts]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the positions of the bars
    x = np.arange(len(edge_counts))
    width = 0.6

    # Create the bar plot for num_ablated_C_gt_M
    ax.bar(x, num_ablated_C_gt_Ms, width, label='Num Ablated C > M', color='b', alpha=0.7)

    # Create horizontal lines for N, N/2, and N/2 ± epsilon * N
    ax.plot(x, ns, label='N', color='k', linestyle='-', linewidth=2)
    ax.plot(x, [n/2 for n in ns], label='N/2', color='r', linestyle='--', linewidth=2)
    ax.plot(x, [n/2 + epsilon*n for n in ns], label=f'N/2 + {epsilon}N', color='m', linestyle=':', linewidth=2)
    ax.plot(x, [n/2 - epsilon*n for n in ns], label=f'N/2 - {epsilon}N', color='c', linestyle=':', linewidth=2)

    # Fill the area between N/2 ± epsilon * N
    ax.fill_between(x, 
                    [n/2 - epsilon*n for n in ns], 
                    [n/2 + epsilon*n for n in ns], 
                    alpha=0.2, color='y', label=f'N/2 ± {epsilon}N range')

    # plot vertical dotted line for min_equiv
    min_equiv_k = next((i for i, k in enumerate(edge_counts) if k == min_equiv), None)
    ax.axvline(x=min_equiv_k, color='g', linestyle='--', label=f'Minimum Equivalent ({min_equiv})')

    # Customize the plot
    ax.set_ylabel('Count')
    ax.set_xlabel('Edge Count')
    ax.set_title(f'Number of Ablated C > M')
    ax.set_xticks(x)
    ax.set_xticklabels(edge_counts, rotation='vertical', fontsize=8)
    ax.legend()

    # Add a grid for better readability
    ax.grid(True, linestyle=':', alpha=0.7)

    # Adjust y-axis to start from 0
    ax.set_ylim(bottom=0)

    # Adjust layout and display the plot
    fig.tight_layout()
    return fig, ax

def plot_circuit_and_model_scores(test_results: Dict[int, EquivResult], min_equiv: int) -> Tuple[plt.Figure, plt.Axes]:
    # mean and std of circ_scores 
    circ_scores_mean = {k: torch.mean(v.circ_scores).cpu() for k, v in test_results.items()}
    circ_scores_std = {k: torch.std(v.circ_scores).cpu() for k, v in test_results.items()}
    # mean and std of model scores 
    model_scores_mean = {k: torch.mean(v.model_scores).cpu() for k, v in test_results.items()}
    model_scores_std = {k: torch.std(v.model_scores).cpu() for k, v in test_results.items()}

    # Convert dictionaries to lists for easier plotting
    labels = list(circ_scores_mean.keys())
    circ_means = list(circ_scores_mean.values())
    circ_stds = list(circ_scores_std.values())

    # Assuming model_scores_mean and model_scores_std are constants
    model_mean = next(iter(model_scores_mean.values())).detach()  # Get the constant model mean
    model_std = next(iter(model_scores_std.values())).detach()  # Get the constant model std

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the positions of the bars
    x = np.arange(len(labels))

    # Create the scatter plot for circuit scores
    x = np.arange(len(labels))
    ax.errorbar(x, circ_means, yerr=circ_stds, fmt='o', capsize=5, label='Circuit Scores')

    # Create horizontal lines for N, N/2, and N/2 ± epsilon * N
    ax.plot(x, [model_mean for _ in labels], label=f'Mean Score (Mean {model_mean:.2f})', color='r', linestyle='--', linewidth=2)
    ax.plot(x, [model_mean + model_std for _ in labels], label=f'Mean + STD', color='m', linestyle=':', linewidth=2)
    ax.plot(x, [model_mean - model_std for _ in labels], label=f'Mean - STD', color='c', linestyle=':', linewidth=2)

    # Fill the area between N/2 ± epsilon * N
    ax.fill_between(x, 
                    [model_mean - model_std for _ in labels], 
                    [model_mean + model_std for _ in labels], 
                    alpha=0.2, color='y', label='Mean ± STD range')

    # Create vertical lines for the minimum equivalent key
    min_equiv_idx = next((i for i, k in enumerate(test_results) if k == min_equiv), None)
    ax.axvline(x=min_equiv_idx, color='g', linestyle='--', label=f'Minimum Equivalent ({min_equiv})')

    # Customize the plot
    ax.set_ylabel('Scores')
    ax.set_xlabel('Edge Count')
    ax.set_title('Circuit Scores vs Constant Model Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical', fontsize=8)
    ax.legend()


    # Adjust layout and display the plot
    fig.tight_layout()
    return fig, ax


def compute_knees(edge_scores):
    # compute knee 
    from kneed import KneeLocator
    import numpy as np
    x = np.linspace(0, len(edge_scores), len(edge_scores))
    y = edge_scores
    kneedle_poly = KneeLocator(x, y, S=1.0, curve="convex", direction="increasing", interp_method="polynomial")
    kneedle_1d = KneeLocator(x, y, S=1.0, curve="convex", direction="increasing", interp_method="interp1d")
    return kneedle_poly, kneedle_1d

def plot_edge_scores_and_knees(edge_scores, kneedle_poly, kneedle_1d, min_equiv):
    fig, ax = plt.subplots()
    ax.plot(edge_scores)
    # log axis 
    ax.set_yscale('log')
    ax.axvline(kneedle_poly.knee, color='r', linestyle='--', label="knee poly")
    ax.axvline(kneedle_1d.knee, color='g', linestyle='--', label="knee 1d")
    # plot min_equiv 
    ax.axvline(len(edge_scores) - min_equiv, color='b', linestyle='--', label="min equiv")
    ax.legend()
    return fig, ax




def create_paths(srcs: set[SrcNode], dests: set[DestNode], n_layers: int) -> list[list[Edge]]:
    #TODO: create paths from src to dest through different to
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

def edges_from_mask(srcs: set[SrcNode], dests: set[DestNode], mask: Dict[str, torch.Tensor], token: bool=False) -> list[Edge]:
    #TODO: fix for SAEs
    SRC_IDX_TO_NODE = {src.src_idx: src for src in srcs}
    DEST_MOD_AND_HEAD_TO_NODE = {(dest.module_name, dest.head_idx): dest for dest in dests}
    edges = []
    for mod_name, mask in mask.items():
        for idx in mask.nonzero():
            # src idx 
            if len(idx) == 1:
                assert not token
                src_idx = idx.item() 
                dest_idx = None
                seq_idx = None
            elif len(idx) == 2 and not token: 
                src_idx = idx[1].item()
                dest_idx = idx[0].item()
                seq_idx = None
            elif len(idx) == 2 and token:
                src_idx = idx[1].item()
                dest_idx = None
                seq_idx = idx[0].item()
            else: 
                assert token and len(idx) == 3
                src_idx = idx[2].item()
                dest_idx = idx[1].item()
                seq_idx = idx[0].item()
            dest_node = DEST_MOD_AND_HEAD_TO_NODE[(mod_name, dest_idx)]
            src_node = SRC_IDX_TO_NODE[src_idx]
            edges.append(Edge(src=src_node, dest=dest_node, seq_idx=seq_idx))
    return edges

def make_complement_paths(
    srcs: set[SrcNode], 
    dests: set[DestNode], 
    n_layers: int, 
    edges: list[Edge],
) -> list[list[Edge]]:
    paths = create_paths(srcs, dests, n_layers)
    edge_set = set(edges)
    complement_paths = [
        path for path in tqdm(paths) if any(edge not in edge_set for edge in path) # new elements in path
    ]
    return complement_paths

def get_edge_idx(edge: Edge, tokens=False):
    # TODO: make backwards compatible
    if edge.dest.name == "Resid End":
        idx = (edge.src.src_idx,)
    elif edge.dest.name.startswith("MLP"):
        idx = (edge.src.src_idx,)
    else:
        idx = (edge.dest.head_idx, edge.src.src_idx)
    if tokens:
        idx = (edge.seq_idx,) + idx
    return idx


def set_score(edge: Edge, scores, value, batch_idx: Optional[int] = None, tokens: bool = False):
    idx = get_edge_idx(edge, tokens=tokens)
    # remove nones
    idx = tuple(filter(lambda x: x is not None, idx))
    # if idx[0] is None:
    #     idx = idx[1:]
    if batch_idx is not None:
        idx = (batch_idx,) + idx
    scores[edge.dest.module_name][idx] = value
    return scores

def join_values(d: Dict[Any, Dict]):
    return {k: v for sub_d in d.values() for k, v in sub_d.items()}

class MinResult(NamedTuple):
    not_minimal: bool
    p_value: float
    num_edge_score_gt_ref: int
    diffs: list[torch.Tensor]
    diffs_inflated: list[torch.Tensor]


def minimality_test( #TODO: seperate infalted circuit seperate from dataset, get higher n 
    model: PatchableModel,
    dataloader: PromptDataLoader,
    attribution_scores: torch.Tensor | PruneScores,
    edges: list[Edge], 
    filtered_paths: list[list[Edge]],
    edge_count: int, 
    ablation_type: AblationType, 
    grad_function: GradFunc,
    answer_function: AnswerFunc,
    circuit_out: Optional[CircuitOutputs] = None,
    threshold: Optional[float] = None,
    use_abs: bool = True,
    tokens: bool = False,
    alpha: float = 0.05,
    q_star: float = 0.9,
) -> Dict[Edge, MinResult]:
    if circuit_out is None:
        circuit_out = dict(next(iter(run_circuits(
            model=model, 
            dataloader=dataloader,
            test_edge_counts=[edge_count],
            prune_scores=attribution_scores,
            patch_type=PatchType.TREE_PATCH,
            ablation_type=ablation_type,
            reverse_clean_corrupt=False,
            use_abs=use_abs
        ).values())))
    if threshold is None:
        threshold = prune_scores_threshold(attribution_scores, edge_count, use_abs=use_abs)
    test_results = {}
    for edge in tqdm(edges):
        test_results[edge] = minimality_test_edge(
            model=model,
            dataloader=dataloader,
            attribution_scores=attribution_scores,
            edge=edge,
            filtered_paths=filtered_paths,
            ablation_type=ablation_type,
            threshold=threshold,
            grad_function=grad_function,
            answer_function=answer_function,
            circuit_out=circuit_out,
            tokens=tokens,
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
    grad_function: GradFunc,
    answer_function: AnswerFunc,
    use_abs: bool = True,  
    tokens: bool = False,
    circuit_out: Optional[CircuitOutputs] = None,
    alpha: float = 0.05,
    q_star: float = 0.9,
) -> MinResult:
    
    # ablate edge and run 
    prune_scores_ablated = deepcopy(attribution_scores)
    prune_scores_ablated[edge.dest.module_name][get_edge_idx(edge, tokens=tokens)] = 0.0
    circuit_out_ablated = join_values(run_circuits(
        model=model, 
        dataloader=dataloader,
        thresholds=[threshold],
        prune_scores=prune_scores_ablated,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
        use_abs=use_abs
    ))

    # TODO: go back over this code, see what's up
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
                set_score(edge, prune_scores_inflated[batch_key], threshold+1, batch_idx=batch_idx, tokens=tokens)
    circuit_out_inflated = join_values(run_circuits(
        model=model, 
        dataloader=dataloader,
        thresholds=[threshold],
        prune_scores=prune_scores_inflated,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
        use_abs=use_abs
    ))

    # ablate random edges in paths and run 
    prune_scores_ablated_paths = prune_scores_inflated
    for batch_key, paths in sampled_paths.items():
        for batch_idx, path in enumerate(paths):
            edge_to_ablate = random.choice(path)
            set_score(edge_to_ablate, prune_scores_ablated_paths[batch_key], 0.0, batch_idx, tokens=tokens)
    circuit_out_ablated_paths = join_values(run_circuits(
        model=model, 
        dataloader=dataloader,
        thresholds=[threshold],
        prune_scores=prune_scores_ablated_paths,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
        reverse_clean_corrupt=False,
        use_abs=use_abs
    ))

    # compute statistics
    n = 0
    num_edge_score_gt_ref = 0
    diffs = []
    diffs_inflated = []
    score_func = get_score_func(grad_function, answer_function)
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
    return MinResult(alpha < p_value, p_value, num_edge_score_gt_ref, diffs, diffs_inflated)

def plot_p_values(min_results: dict[Edge, MinResult], edges: list[Edge], edge_scores: dict[Edge, torch.Tensor]):
    fig, ax = plt.subplots(figsize=(12, 2))
    p_values = [min_results[edge].p_value for edge in edges]
    neg_edge = [edge_scores[edge].cpu() < 0 for edge in edges]
    ax.scatter(range(len(p_values)), p_values, c=neg_edge, cmap='coolwarm')
    # plot alpha line 
    ax.axhline(y=0.05, color='g', linestyle='-')
    ax.set_title("p values for minimality test")
    return fig, ax

def plot_edge_k(min_results: dict[Edge, MinResult], edges: list[Edge], edge_scores: dict[Edge, torch.Tensor], n: int, q_star: float):
    fig, ax = plt.subplots(figsize=(12, 2))
    ks = [min_results[edge][2] for edge in edges]
    neg_edge = [edge_scores[edge].cpu() < 0 for edge in edges]
    # scatter with blue as positive, red as negative
    ax.scatter(range(len(ks)), ks, c=neg_edge, cmap='coolwarm')
    # horizontal line at  
    ax.axhline(y=n // 2, color='g', linestyle='--', label=f"N / 2")
    # horizeontal line at n * q_star
    ax.axhline(y=n * q_star, color='r', linestyle='--', label=f"N x q* ({q_star})")

    ax.set_title("k for minimality test")

    ax.legend()
    return fig, ax

def plot_score_quantiles(
    min_results: dict[Edge, MinResult],
    edges: list[Edge],
    edge_scores: dict[Edge, torch.Tensor],
    quantile_range: list[float] = [0.00, 1.00]
):
    # calculate quantiles 
    quantiles = [
        np.quantile(torch.cat(min_results[edge][3]).detach().cpu().numpy(), quantile_range) 
        for edge in edges
    ]
    lower_quantiles = [q[0] for q in quantiles]
    upper_quantiles = [q[1] for q in quantiles]

    # compute mean and quartiles of diff inflated
    diff_infl = torch.cat([torch.cat(min_results[edge][4]) for edge in edges])
    quantile_infl = np.quantile(diff_infl.detach().cpu().numpy(), quantile_range)
    mean_infl = diff_infl.mean().detach().cpu().numpy()
    median_infl = diff_infl.median().detach().cpu().numpy()

    # plot average diff with quantile ranges
    fig, ax = plt.subplots(figsize=(12, 4))
    diffs = [torch.cat(min_results[edge][3]).mean().detach().cpu().numpy() for edge in edges]
    median_diffs = [torch.cat(min_results[edge][3]).median().detach().cpu().numpy() for edge in edges]

    # Plot error bars with quantile ranges, median, and mean
    ax.errorbar(range(len(diffs)), diffs, 
                yerr=[np.array(diffs) - lower_quantiles, upper_quantiles - np.array(diffs)],
                fmt='none', capsize=5, capthick=1)

    # Add median points in orange
    ax.scatter(range(len(median_diffs)), median_diffs, color='orange', marker='s', s=30, label='Median', zorder=3)

    # Add mean points in green
    ax.scatter(range(len(diffs)), diffs, color='green', marker='o', s=30, label='Mean', zorder=3)

    # inflated mean and median lines
    ax.axhline(y=mean_infl, color='g', linestyle='-')
    ax.axhline(y=median_infl, color='orange', linestyle='-')

    # Add quantile inflation lines
    ax.axhline(y=quantile_infl[0], color='c', linestyle='--',  zorder=2, label=f'Inflated Quantile Range ({quantile_range[0]*100})')
    ax.axhline(y=quantile_infl[1], color='m', linestyle='--', zorder=2, label=f'Inflated Quantile Range ({quantile_range[1]*100})')

    ax.set_yscale('log')
    ax.set_title(f"Score diff for minimality test (with {quantile_range[0]*100}-{quantile_range[1]*100} quantile ranges)")
    ax.set_xlabel("Edges")
    ax.set_ylabel("Score Difference")

    # Add legend
    ax.legend()
    return fig, ax
    



