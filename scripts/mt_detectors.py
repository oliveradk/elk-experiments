import os
import submitit
from datetime import datetime as dt
os.environ["HF_HOME"] = "/nas/ucb/oliveradk/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/nas/ucb/oliveradk/.cache"

exp_dir = f"/nas/ucb/oliveradk/elk-experiments/output/{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}"
def train():
    import torch
    from cupbearer import utils, data, detectors, tasks, models, scripts

    task = tasks.measurement_tampering(task_name="diamonds", device="cuda", untrusted_labels=True)

    cache = detectors.ActivationCache()
    # cache = detectors.ActivationCache.load("/nas/ucb/oliveradk/elk-experiments/output/2024-05-27_20-27-47/cache.pt")
    # cache = None

    names = [
        # "hf_model.transformer.h.10.output",
        # "hf_model.transformer.h.19.ouptut",
        "hf_model.transformer.ln_f.output"

    ]

    def get_activation_at_last_token(
        activation: torch.Tensor | tuple, inputs: list[str], name: str
    ):
        if isinstance(activation, tuple):
            activation = activation[0]
        assert activation.shape[1] == 1024
        return activation[:, -1, :]
    
    def get_activations_at_sensor_tokens(
        activation: torch.Tensor, inputs: list[str], name: str,
    ):
        if isinstance(activation, tuple):
            activation = activation[0]
        tokens = task.model.tokenize(inputs, **task.model.tokenize_kwargs)["input_ids"] # batch size x seq len
        flat_tensor_token_idxs = (tokens == task.model.hf_model.sensor_token_id).nonzero(as_tuple=True)[1]
        tensor_token_idxs = flat_tensor_token_idxs.view(-1, task.model.hf_model.n_sensors)
        sensor_acts = activation.gather(
            1, tensor_token_idxs.unsqueeze(-1).expand(-1, -1, task.model.hf_model.config.emb_dim)
        )
        last_act = activation[:, -1, :].unsqueeze(1)
        sensor_and_last_acts = torch.concat([sensor_acts, last_act], dim=1)
        return sensor_and_last_acts

    # gt_detector = detectors.SupervisedLinearProbe(
    #     names, activation_processing_func=get_activation_at_last_token,
    #     layer_aggregation="mean", cache=cache
    # )

    # scripts.train_detector(
    #     task, gt_detector, save_path=exp_dir + "/" + "gt", eval_batch_size=8, batch_size=8, max_iter=100 # more iterations overfits I think?
    # )

    # cache.store(exp_dir + "/" + "cache")

    detector = detectors.MahalanobisDetector(
        activation_names=names, activation_processing_func=get_activations_at_sensor_tokens,#get_activation_at_last_token,
        layer_aggregation="mean", cache=cache
    )

    scripts.train_detector(
        task, detector, save_path=exp_dir + "/" +  "mahalanobis", eval_batch_size=8, batch_size=8
    ) 


slurm_params = {
        "slurm_mem_gb": 80, 
        "slurm_gres": "gpu:A100-SXM4-80GB:1",
        "nodes": 1, 
        "timeout_min": 60 * 10,
        "slurm_job_name": "bash",
        "slurm_qos": "high",
#         "additional_parameters":{
#             'export': 'HF_HOME="/nas/ucb/oliveradk/.cache",TRANSFORMERS_CACHE="/nas/ucb/oliveradk/.cache"'
#         }
}
executor = submitit.AutoExecutor(folder=exp_dir)
executor.update_parameters(**slurm_params)

job = executor.submit(train)