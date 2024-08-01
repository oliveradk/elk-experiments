from pathlib import Path
from typing import Tuple, Callable, Any, Dict, List, Set
from abc import abstractmethod, ABC
from functools import partial
from collections import defaultdict
from tqdm import tqdm

import torch 

from transformer_lens import HookedTransformer

from cupbearer import utils
from cupbearer.detectors import AnomalyDetector,  ActivationCache
from cupbearer.detectors.anomaly_detector import plot_scores
from cupbearer.detectors.statistical import StatisticalDetector
from cupbearer.data import MixedData


from auto_circuit.data import PromptDataset, PromptDataLoader, PromptPair, PromptPairBatch
from auto_circuit.types import AblationType, PatchType, PruneScores, CircuitOutputs
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model, train_mask_mode, set_all_masks
from auto_circuit.utils.tensor_ops import desc_prune_scores, prune_scores_threshold, batch_avg_answer_diff
from auto_circuit.utils.patch_wrapper import PatchWrapperImpl
from auto_circuit.utils.misc import module_by_name

from elk_experiments.utils import set_model
from elk_experiments.auto_circuit.auto_circuit_utils import (
    make_prompt_data_loader,
    make_mixed_prompt_dataloader,
)




class AutoCircuitDetector(AnomalyDetector):

    def __init__(
        self, 
        effect_tokens: list[int], 
        device: str = "cpu", 
        resid_src=False, 
        resid_dest=False,
        attn_src=True,
        attn_dest=True,
        mlp_src=True,
        mlp_dest=True,
        **kwargs
    ):
        self.effect_tokens = effect_tokens
        self.resid_src = resid_src
        self.resid_dest = resid_dest
        self.attn_src = attn_src
        self.attn_dest = attn_dest
        self.mlp_src = mlp_src
        self.mlp_dest = mlp_dest
        self.device = device
        super().__init__(**kwargs)

    def set_model(self, model: HookedTransformer):
        # apply tl hooks
        set_model(model)
        # apply patching wrapper
        self.model = patchable_model(
            model, 
            factorized=True, 
            slice_output="last_seq",
            separate_qkv=True, 
            resid_src=self.resid_src,
            resid_dest=self.resid_dest,
            attn_src=self.attn_src,
            attn_dest=self.attn_dest,
            mlp_src=self.mlp_src,
            mlp_dest=self.mlp_dest,
            device=self.device
        )


class AutoCircuitPruningDetector(AutoCircuitDetector):
    def __init__(
        self, 
        effect_tokens: list[int],
        prune_scores_func: Callable[[HookedTransformer, PromptDataLoader], Tuple[PruneScores, CircuitOutputs]], 
        scores_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        patch_type: PatchType = PatchType.TREE_PATCH,
        k: int | None = None, 
        threshold: float | None = None,
        device: str = "cpu",
        layer_aggregation: str = "mean",
        **kwargs
        #TODO: option to use untrusted data
    ): 
        assert (k is not None) ^ (threshold is not None), "Either k or threshold must be specified"
        self.prune_scores_func = prune_scores_func
        self.scores_func = scores_func
        self.patch_type = patch_type    
        self.k = k
        self.threshold = threshold
        super().__init__(
            effect_tokens=effect_tokens,
            device=device, 
            layer_aggregation=layer_aggregation,
            **kwargs
        )

    def train(
        self,
        trusted_data,
        untrusted_data,
        save_path: Path | str,
        *,
        batch_size: int = 64,
        **trainer_kwargs,
    ):
        # construct prompt datasets and loaders
        trusted_dataloader = make_prompt_data_loader(trusted_data, self.effect_tokens, self.model, batch_size)
        self.pruning_scores, self.patch_src_outs = self.prune_scores_func(
            model=self.model,
            dataloader=trusted_dataloader
        )
        #TODO: option to use untrusted data
    
    def eval(
        self,
        dataset: MixedData, 
        batch_size: int = 1024,
        histogram_percentile: float = 95,
        save_path: Path | str | None = None,
        num_bins: int = 100,
        pbar: bool = False,
        layerwise: bool = False,
        log_yaxis: bool = True,
    ):
        dataloader = make_mixed_prompt_dataloader(dataset, self.effect_tokens, self.model, batch_size)
        return super().eval(
            dataset=None,
            test_loader=dataloader,
            batch_size=batch_size,
            histogram_percentile=histogram_percentile,
            save_path=save_path,
            num_bins=num_bins,
            pbar=pbar,
            layerwise=layerwise,
            log_yaxis=log_yaxis,
        )

    def _get_trained_variables(self, saving: bool = False):
        return {
            "pruning_scores": self.pruning_scores,
            "patch_src_outs": self.patch_src_outs
        }.update(super()._get_trained_variables(saving))
    
    
    def _set_trained_variables(self, variables):
        self.pruning_scores = variables["pruning_scores"]
        self.patch_src_outs = variables["patch_src_outs"]
        super()._set_trained_variables(variables)

    
    def layerwise_scores(self, batch):
        raise NotImplementedError(
            "Layerwise scores don't exist for finetuning detector"
        )

    def scores(self, batch):
        # for now, assume we're reusing the patch src outs, mean over tokens
        input = utils.inputs_from_batch(batch)
        input = input.clean
        
        desc_ps = desc_prune_scores(self.pruning_scores) #TODO: look back at the other method they had
        threshold = prune_scores_threshold(desc_ps, self.k) if self.threshold is None else self.threshold

        # run patched model 
        with patch_mode(self.model, self.patch_src_outs):
            # set patch masks according to threshold
            patch_edge_count = 0 
            for mod_name, patch_mask in self.pruning_scores.items():
                dest = module_by_name(self.model, mod_name)
                assert isinstance(dest, PatchWrapperImpl)
                assert dest.is_dest and dest.patch_mask is not None 
                if self.patch_type == PatchType.EDGE_PATCH: # patch out edges in circuit
                    dest.patch_mask.data = (patch_mask.abs() >= threshold).float()
                    patch_edge_count += dest.patch_mask.int().sum().item()
                else:
                    assert self.patch_type == PatchType.TREE_PATCH # patch out edges not in circuit
                    dest.patch_mask.data = (patch_mask.abs() < threshold).float()
                    patch_edge_count += (1 - dest.patch_mask.int()).sum().item()
            # run model
            with torch.inference_mode():
                patched_logits = self.model(input)[self.model.out_slice]

        # normal output 
        with torch.inference_mode():
            normal_logits = self.model(input)[self.model.out_slice]
        
        # compute scores
        scores = self.scores_func(patched_logits, normal_logits)
        if self.patch_type == PatchType.EDGE_PATCH: # invert scores if patching circuit
            scores = - scores
        return scores


class AutoCircuitGradScoresDetector(AutoCircuitDetector, StatisticalDetector):

    def __init__(
        self, 
        effect_tokens: list[int],
        ablation_type: AblationType = AblationType.ZERO,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = batch_avg_answer_diff,
        ignored_layers: Set[int] = {},
        device: str = "cpu",
        cache: ActivationCache | None = None,
        layer_aggregation: str = "mean",
        **kwargs
    ):
        assert ablation_type.mean_over_dataset or ablation_type.ZERO, "Only mean over dataset and zero ablation supported"
        self.ablation_type = ablation_type
        self.metric = metric
        self.patch_outs: torch.Tensor = None
        self.ignored_layers = ignored_layers
        super().__init__(
            effect_tokens=effect_tokens,
            device=device,
            cache=cache,
            activation_names=["prune_scores"],
            layer_aggregation=layer_aggregation,
            **kwargs
        )

    def compute_patch_outs(self, data_loader: PromptDataLoader):
        self.reset_patch_outs()
        # pass dataloader b/c patch outs must be consistent across instances, batches
        # e.g. mean ablation or zero ablation
        if self.ablation_type.mean_over_dataset:
            sample = data_loader 
        elif self.ablation_type == AblationType.ZERO:
            sample = next(iter(data_loader)).clean
        else:
            raise NotImplementedError(self.ablation_type)
        patch_outs = src_ablations(self.model, sample, self.ablation_type)
        self.patch_outs = patch_outs.clone().detach()
   
    
    def reset_patch_outs(self):
        self.patch_outs = None
    
    def train(
        self,
        trusted_data,
        untrusted_data,
        *,
        batch_size: int = 1024,
        pbar: bool = True,
        max_steps: int | None = None,
        **kwargs,
    ):
        # NOTE: a lot of this is copied from the default train method in StatisticalDetector
        # but didn't want to add too many hooks, so fine for now
        
        # It's important we don't use torch.inference_mode() here, since we want
        # to be able to override this in certain detectors using torch.enable_grad().
        if self.use_trusted:
            if trusted_data is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires trusted training data."
                )
            data = trusted_data
        else:
            if untrusted_data is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires untrusted training data."
                )
            data = untrusted_data
        data_loader = make_prompt_data_loader(data, self.effect_tokens, self.model, batch_size)
        
        # init statistical variables
        #TODO: will need to change when filtering is added
        activation_sizes = {"prune_scores": torch.Size((self.model.n_edges,))}
        self.init_variables(activation_sizes, device=self.device)

        # compute patch outs (consistent acorss instances, batches)
        self.compute_patch_outs(data_loader)

        if pbar:
            data_loader = tqdm(data_loader, total=max_steps or len(data_loader))

        # compute prune scores
        #TODO: add filtering 
        with train_mask_mode(self.model):
            set_all_masks(self.model, val=0.0)
            for batch in data_loader:
                # get activations (prune scores)
                activations = self.get_activations(batch)
                self.model.zero_grad()
                self.batch_update(activations)
        # remove patch outs to save memory
        # self.reset_patch_outs()
        print("leaving auto_circuit train")
    
    def eval(
        self,
        dataset: MixedData, 
        batch_size: int = 1024,
        histogram_percentile: float = 95,
        save_path: Path | str | None = None,
        num_bins: int = 100,
        pbar: bool = False,
        layerwise: bool = False,
        log_yaxis: bool = True,
    ):
        assert 0 < histogram_percentile <= 100

        test_loader = make_mixed_prompt_dataloader(dataset, self.effect_tokens, self.model, batch_size)

        if pbar:
            test_loader = tqdm(test_loader, desc="Evaluating", leave=False)

        scores = defaultdict(list)
        labels = defaultdict(list)

        # NOTE: this is not very DRY - (copied from AnomalyDetector eval with extra context handlers and set all_masks)
        # could add optional context handlers and hooks to default eval,  but for now this is fine 
        with train_mask_mode(self.model):
            set_all_masks(self.model, val=0.0)
            for batch in test_loader:
                inputs, new_labels = batch
                if layerwise:
                    new_scores = self.layerwise_scores(inputs)
                else:
                    new_scores = {"all": self.scores(inputs)}
                self.model.zero_grad()
                for layer, score in new_scores.items():
                    if isinstance(score, torch.Tensor):
                        score = score.cpu().numpy()
                    assert score.shape == new_labels.shape
                    scores[layer].append(score)
                    labels[layer].append(new_labels)
        
        return plot_scores(
            scores, labels, histogram_percentile, num_bins, log_yaxis, save_path
        )
    
    def _get_activations_no_cache(self, inputs: PromptPairBatch) -> dict[str, torch.Tensor]:
        # raise warning if not called within the proper context
        if not next(iter(self.model.dest_wrappers)).patch_mask.requires_grad:
            raise RuntimeError("Model must be in train_mask_mode to compute activations")
        
        with patch_mode(self.model, self.patch_outs, batch_size=inputs.clean.shape[0]):
            logits = self.model(inputs.clean)
            loss = -self.metric(logits, inputs=inputs)
            loss.backward()
        prune_scores = {
            dest_wrapper.module_name: dest_wrapper.patch_mask_batch.grad.detach().clone()
            for dest_wrapper in self.model.dest_wrappers
        }
        # filter out dest nodes 
        # filter out indices of src nodes from each
        prune_scores_vec = torch.concat([v.flatten(start_dim=1) for v in prune_scores.values()], dim=1)
        assert prune_scores_vec.shape == (inputs.clean.shape[0], self.model.n_edges)
        return {"prune_scores": prune_scores_vec} # TODO: could return individual scores for each destination

                

