import torch
from abc import ABC
from functools import partial
from torch import Tensor

from cupbearer import utils
from cupbearer.detectors.statistical import StatisticalDetector, MahalanobisDetector

from eap.eap_wrapper import EAP_clean_forward_hook, EAP_clean_backward_hook
from eap.eap_graph import EAPGraph



def get_edge_scores_filter_indices(edge_filter, graph):
    valid_upstream_idxs = [i for (i, node) in enumerate(graph.upstream_nodes) if edge_filter(node)]
    valid_downstream_idxs = [i for (i, node) in enumerate(graph.downstream_nodes) if edge_filter(node)]
    uu, dd = torch.meshgrid(torch.tensor(valid_upstream_idxs), torch.tensor(valid_downstream_idxs), indexing='ij')
    return uu, dd



from transformer_lens import HookedTransformer

class EAPDetector(StatisticalDetector, ABC):

    EAP_SCORES_NAME = "eap_scores"
    
    def __init__(
          self, 
          effect_prob_func, 
          upstream_nodes=["head", "mlp"], 
          downstream_nodes=["head", "mlp"],
          edge_filter=lambda x : True,
          seq_len=16, # would ideally pass on train and eval, but don't want to change script (shrug)
          **kwargs
    ):
        self.effect_prob_func = effect_prob_func
        self.upstream_nodes = upstream_nodes
        self.downstream_nodes = downstream_nodes
        self.edge_filter = edge_filter
        self.seq_len = seq_len 
        self.trusted_graph = None 
        self.untrusted_graphs = []
        super().__init__(
            activation_names=[self.EAP_SCORES_NAME], 
            activation_processing_func=lambda x: x,
            **kwargs
        )
    
    def set_model(self, model: HookedTransformer):
        super().set_model(model)
        self.graph = EAPGraph(model.cfg, self.upstream_nodes, self.downstream_nodes, aggregate_batch=False, verbose=False)
        self.uu, self.dd = get_edge_scores_filter_indices(self.edge_filter, self.graph)
    
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
        with torch.enable_grad():
            inputs = utils.inputs_to_device(inputs, self.model.cfg.device)
            batch_size, seq_len = inputs.shape[:2]
            self.model.reset_hooks()
            self._set_hooks(batch_size, seq_len=seq_len)
            #TODO: add support for corrupted tokens
            self.model.add_hook(self.upstream_hook_filter, self.clean_upstream_hook_fn, "fwd")
            self.model.add_hook(self.downstream_hook_filter, self.clean_downstream_hook_fn, "bwd")

            value = self.effect_prob_func(self.model(inputs, return_type="logits"))
            value.backward()

            self.model.zero_grad()
            self.upstream_activations_difference *= 0
            return {self.EAP_SCORES_NAME: self.graph.eap_scores[:, self.uu, self.dd]}
    

class EAPMahalanobisDetector(EAPDetector, MahalanobisDetector):
    pass