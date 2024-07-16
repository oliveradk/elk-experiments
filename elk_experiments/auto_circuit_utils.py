
from typing import List, Tuple

import torch 
from cupbearer.data import MixedData

from auto_circuit.data import PromptDataset, PromptDataLoader, PromptPair, PromptPairBatch
from auto_circuit.types import AblationType, PatchType, PruneScores, CircuitOutputs
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model, train_mask_mode, set_all_masks, PatchableModel
from auto_circuit.utils.tensor_ops import desc_prune_scores, prune_scores_threshold, batch_avg_answer_diff
from auto_circuit.utils.patch_wrapper import PatchWrapperImpl
from auto_circuit.utils.misc import module_by_name



def make_prompt_dataset(data, effect_tokens, vocab_size):
    clean_prompts = [x[0] for x in data]
    answers = [effect_tokens] * len(clean_prompts)
    wrong_answers = [list(set(range(vocab_size)) - set(answer)) for answer in answers]
    
    # put into torch tensors
    clean_prompts = torch.stack(clean_prompts, dim=0)
    corrupt_prompts = torch.stack([torch.zeros_like(clean_prompts[0], dtype=int)] * len(clean_prompts), dim=0)
    answers = [torch.tensor(answer, dtype=int) for answer in answers]
    wrong_answers= [torch.tensor(answer, dtype=int) for answer in wrong_answers]

    return PromptDataset(clean_prompts, corrupt_prompts, answers, wrong_answers)

def make_mixed_prompt_dataloader(dataset: MixedData, effect_tokens, model, batch_size):
    normal_dataset = make_prompt_dataset(dataset.normal_data, effect_tokens, model.tokenizer.vocab_size)
    anomalous_dataset = make_prompt_dataset(dataset.anomalous_data, effect_tokens, model.tokenizer.vocab_size)
    prompt_dataset = MixedData(normal_dataset, anomalous_dataset, dataset.normal_weight, dataset.return_anomaly_labels)
    seq_len = normal_dataset.clean_prompts.size(1)
    dataloader = PromptDataLoader(
        prompt_dataset,
        seq_len=seq_len,
        diverge_idx=0,
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_fn
    )
    return dataloader

def make_prompt_data_loader(dataset, effect_tokens, model, batch_size):
    prompt_dataset = make_prompt_dataset(dataset, effect_tokens, model.tokenizer.vocab_size)
    seq_len = prompt_dataset.clean_prompts.size(1)
    dataloader = PromptDataLoader(
        prompt_dataset,
        seq_len=seq_len,
        diverge_idx=0,
        batch_size=batch_size,
        shuffle=False
    )
    return dataloader



def collate_fn(batch: List[Tuple[PromptPair, int]]) -> Tuple[PromptPairBatch, torch.Tensor]:
    clean = torch.stack([p.clean for p, y in batch])
    corrupt = torch.stack([p.corrupt for p, y in batch])
    labels = torch.tensor([y for p, y in batch], dtype=torch.int)
    if all([p.answers.shape == batch[0][0].answers.shape for p, y in batch]):
        answers = torch.stack([p.answers for p, y in batch])
    else:  # Sometimes each prompt has a different number of answers
        answers = [p.answers for p, y in batch]
    if all([p.wrong_answers.shape == batch[0][0].wrong_answers.shape for p, y in batch]):
        wrong_answers = torch.stack([p.wrong_answers for p, y in batch])
    else:  # Sometimes each prompt has a different number of wrong answers
        wrong_answers = [p.wrong_answers for p, y in batch]
    key = hash((str(clean.tolist()), str(corrupt.tolist())))

    diverge_idxs = (~(clean == corrupt)).int().argmax(dim=1)
    batch_dvrg_idx: int = int(diverge_idxs.min().item())
    return PromptPairBatch(key, batch_dvrg_idx, clean, corrupt, answers, wrong_answers), labels


def sorted_scores(scores: PruneScores, model: PatchableModel):
    sorted_srcs = sorted(model.srcs, key=lambda x: x.src_idx)
    sorted_dests_by_module = {mod.module_name: sorted([dest for dest in model.dests if dest.module_name == mod.module_name], key=lambda x: x.head_idx) for mod in model.dest_wrappers}
    
    score_thruple = []
    for mod_name, score in scores.items():
        if score.ndim == 1:
            score = score.unsqueeze(0)
        for i in range(score.shape[0]):
            dest_name = sorted_dests_by_module[mod_name][i].name
            for j in range(score.shape[1]):
                src_name = sorted_srcs[j].name
                score_thruple.append((src_name, dest_name, score[i, j]))
    score_thruple.sort(key=lambda x: abs(x[2]), reverse=True)
    return score_thruple