import torch
from abc import ABC
from functools import partial

import numpy as np
from torch import Tensor

from transformer_lens import HookedTransformer

from cupbearer import utils
from cupbearer.detectors import ActivationBasedDetector

from eap.eap_wrapper import EAP_clean_forward_hook, EAP_clean_backward_hook
from eap.eap_graph import EAPGraph

def node_layer(node):
    return int(node.split(".")[1])

#TODO: put in EAPGraph

def valid_edge(edge: tuple[str, str]):
    return node_layer(edge[0]) <= node_layer(edge[1])

def layer_edge_filter(edge, excluded_layers={}):
    return (node_layer(edge[0]) not in excluded_layers) and (node_layer(edge[1]) not in excluded_layers)

def effect_prob_func(logits, effect_tokens=None, other_tokens=None, inputs=None):
    assert logits.ndim == 3
    assert effect_tokens is not None
    # Sum over vocab and batch dim (for now we're just computing attribution values, we'll deal with per data instance later)
    last_logits = logits[:, -1, :]
    out = last_logits[:, effect_tokens].mean(dim=-1)
    if other_tokens is not None:
        out -= last_logits[:, other_tokens].mean(dim=-1)
    print(out)
    out = out.sum() # mean over batch
    # out = logits[:, -1, effect_tokens].mean()
    return out

def set_model(model: HookedTransformer):
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    


class EAPDetector(ActivationBasedDetector, ABC):

    EAP_SCORES_NAME = "eap_scores"
    
    def __init__(
          self, 
          effect_prob_func, 
          upstream_nodes=["head", "mlp"], 
          downstream_nodes=["head", "mlp"],
          edge_filter=lambda x : True,
          input_filter = lambda x: x,
          **kwargs
    ):
        self.effect_prob_func = effect_prob_func
        self.upstream_nodes = upstream_nodes
        self.downstream_nodes = downstream_nodes
        self.edge_filter = edge_filter
        self.input_filter = input_filter
        self.trusted_graph = None 
        self.untrusted_graphs = []
        self.edge_names_arr = None
        super().__init__(
            activation_names=[self.EAP_SCORES_NAME], 
            activation_processing_func=lambda x: x,
            **kwargs
        )

    def set_graph(self, model):
        self.graph = EAPGraph(model.cfg, self.upstream_nodes, self.downstream_nodes, aggregate_batch=False, verbose=False)
        if self.edge_names_arr is None:
            self.edge_names_arr = np.array([
                [(upstream_node, downstream_node) for downstream_node in self.graph.downstream_nodes]
                for upstream_node in self.graph.upstream_nodes]
            )
        # valid edge mask
        valid_edge_mask = np.apply_along_axis(valid_edge, 2, self.edge_names_arr)
        # edge filter mask
        edge_filter_mask = np.apply_along_axis(self.edge_filter, 2, self.edge_names_arr)

        self.edge_mask = valid_edge_mask & edge_filter_mask
        self.num_valid_edges = self.edge_mask.sum()
    
    def set_model(self, model: HookedTransformer):
        super().set_model(model)
        self.set_graph(model)
    
    def _set_hooks(self, batch_size, seq_len):
        # import ipdb; ipdb.set_trace()
        self.upstream_activations_difference = torch.zeros(
            (batch_size, seq_len, self.graph.n_upstream_nodes, self.model.cfg.d_model),
            device=self.model.cfg.device,
            dtype=self.model.cfg.dtype,
            requires_grad=False
        )

        # set the EAP scores to zero
        self.graph.reset_scores(batch_size)

        self.upstream_hook_filter = lambda name: name.endswith(tuple(self.graph.upstream_hooks))
        self.downstream_hook_filter = lambda name: name.endswith(tuple(self.graph.downstream_hooks))

        self.clean_upstream_hook_fn = partial(
            EAP_clean_forward_hook,
            upstream_activations_difference=self.upstream_activations_difference,
            graph=self.graph
        )

        self.clean_downstream_hook_fn = partial(
            EAP_clean_backward_hook,
            upstream_activations_difference=self.upstream_activations_difference,
            graph=self.graph, 
            aggregate_batch=False
        )

    def _get_activations_no_cache(self, inputs) -> dict[str, Tensor]:
        # TODO: include competion token
        with torch.enable_grad():
            inputs = utils.inputs_to_device(inputs, self.model.cfg.device)
            model_input = self.input_filter(inputs)
            batch_size, seq_len = model_input.shape[:2]
            self.model.reset_hooks()
            self._set_hooks(batch_size, seq_len=seq_len)
            #TODO: add support for corrupted tokens
            self.model.add_hook(self.upstream_hook_filter, self.clean_upstream_hook_fn, "fwd")
            self.model.add_hook(self.downstream_hook_filter, self.clean_downstream_hook_fn, "bwd")

            value = self.effect_prob_func(self.model(model_input, return_type="logits"), inputs=inputs)
            value.backward()

            self.model.zero_grad()
            self.upstream_activations_difference *= 0
            eap_scores_flat = self.graph.eap_scores[:, self.edge_mask]
            assert eap_scores_flat.shape == (batch_size, self.num_valid_edges)
            return {self.EAP_SCORES_NAME: eap_scores_flat}