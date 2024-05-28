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
        "hf_model.transformer.ln_f.output"
    ]

    def get_activation_at_last_token(
        activation: torch.Tensor, inputs: list[str], name: str
    ):
        assert activation.shape[1] == 1024
        return activation[:, -1, :]

    gt_detector = detectors.SupervisedLinearProbe(
        names, activation_processing_func=get_activation_at_last_token,
        layer_aggregation="mean", cache=cache
    )

    scripts.train_detector(
        task, gt_detector, save_path=exp_dir + "/" + "gt", eval_batch_size=8, batch_size=8, max_iter=100 # more iterations overfits I think?
    )

    # cache.store(exp_dir + "/" + "cache")

    detector = detectors.MahalanobisDetector(
        activation_names=names, activation_processing_func=get_activation_at_last_token,
        layer_aggregation="mean", cache=cache
    )

    scripts.train_detector(
        task, detector, save_path=exp_dir + "/" +  "mahalanobis", eval_batch_size=4, batch_size=4
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