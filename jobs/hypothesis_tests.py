import itertools
from datetime import datetime

import submitit 

from elk_experiments.utils import repo_path_to_abs_path, OUTPUT_DIR
from auto_circuit.tasks import (
    IOI_TOKEN_CIRCUIT_TASK, 
    IOI_COMPONENT_CIRCUIT_TASK, 
    IOI_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK, 
    DOCSTRING_TOKEN_CIRCUIT_TASK, 
    DOCSTRING_COMPONENT_CIRCUIT_TASK, 
    GREATERTHAN_COMPONENT_CIRCUIT_TASK, 
    GREATERTHAN_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK, 
    ANIMAL_DIET_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK
)
from auto_circuit.types import AblationType
from elk_experiments.auto_circuit.score_funcs import GradFunc, AnswerFunc

#specify the hyperparameters
tasks = [
    IOI_TOKEN_CIRCUIT_TASK, 
    IOI_COMPONENT_CIRCUIT_TASK, 
    DOCSTRING_TOKEN_CIRCUIT_TASK, 
    DOCSTRING_COMPONENT_CIRCUIT_TASK, 
    GREATERTHAN_COMPONENT_CIRCUIT_TASK, 
    # IOI_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK, 
    # GREATERTHAN_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK, 
    # ANIMAL_DIET_GPT2_AUTOENCODER_COMPONENT_CIRCUIT_TASK
]
use_abs = [True, False]
ablation_types = [
    AblationType.RESAMPLE, 
    AblationType.TOKENWISE_MEAN_CLEAN,
    AblationType.TOKENWISE_MEAN_CORRUPT, 
    AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT,
    AblationType.ZERO
]
grad_funcs = [
    GradFunc.LOGIT, 
    GradFunc.LOGPROB
]
answer_funcs = [
    AnswerFunc.MAX_DIFF, 
    AnswerFunc.AVG_VAL
]

use_abs_to_epsilon = {
    True: [0.1], 
    False: [0.0, -0.4]
}

# generate all combinations
combinations = [
    {
        "task": f"'{task.key}'",
        "use_abs": use_ab,
        "ablation_type": ablation_type.name,
        "grad_func": grad_func.name,
        "answer_func": answer_func.name,
        "epsilon": epsilon
    }
    for task, use_ab, ablation_type, grad_func, answer_func in itertools.product(
        tasks,
        use_abs,
        ablation_types,
        grad_funcs,
        answer_funcs
    )
    for epsilon in use_abs_to_epsilon[use_ab]
]

# combinations = [
#     {
#         "task": f"'{IOI_TOKEN_CIRCUIT_TASK.key}'",
#         "use_abs": False,
#         "ablation_type": AblationType.TOKENWISE_MEAN_CORRUPT.name,
#         "grad_func": GradFunc.LOGPROB.name,
#         "answer_func": AnswerFunc.AVG_VAL.name,
#         "epsilon": 0.0
#     }
# ]

# setup the executor
out_dir = repo_path_to_abs_path(OUTPUT_DIR / "hypo_test_out_logs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
out_dir.mkdir(exist_ok=True, parents=True)
executor = submitit.AutoExecutor(folder=out_dir)
executor.update_parameters(
    timeout_min=60*24,
    mem_gb=40,
    gres="gpu:1",
    cpus_per_task=8,
    nodes=1,
    slurm_qos="high", 
    slurm_array_parallelism=4
)
# run the jobs
with executor.batch():
    for combo in combinations:
        function = submitit.helpers.CommandFunction(
            ["python", "scripts/hypothesis_tests.py"] + [
                f"{key}={value}" for key, value in combo.items()
            ], 
            cwd=repo_path_to_abs_path(".")
        )
        executor.submit(function)



