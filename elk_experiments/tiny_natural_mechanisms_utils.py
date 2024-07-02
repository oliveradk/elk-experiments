from cupbearer import tasks
from cupbearer.tasks.tiny_natural_mechanisms import TinyNaturalMechanismsDataset
def get_task_subset(task, trusted_subset=None, normal_test_subset=None, anomalous_test_subset=None):
    # set subsets to full dataset if not provided
    trusted_subset = trusted_subset if trusted_subset is not None else len(task.trusted_data.data)
    normal_test_subset = normal_test_subset if normal_test_subset is not None else len(task.test_data.normal_data.data)
    anomalous_test_subset = anomalous_test_subset if anomalous_test_subset is not None else len(task.test_data.anomalous_data.data)
    # get subsets
    trusted_data = task.trusted_data.data[:trusted_subset]
    normal_test_data = task.test_data.normal_data.data[:normal_test_subset]
    anomalous_test_data = task.test_data.anomalous_data.data[:anomalous_test_subset]
    # construct new task with subsets
    return tasks.Task.from_separate_data(
        model=task.model, 
        trusted_data=TinyNaturalMechanismsDataset(trusted_data),
        clean_test_data=TinyNaturalMechanismsDataset(normal_test_data), 
        anomalous_test_data=TinyNaturalMechanismsDataset(anomalous_test_data)
    )

def dataset_to_dict(dataset):
    return [{"prefix_tokens": x[0], "completion_token": x[1]} for x in dataset]

def task_data_to_dicts(task):
    return tasks.Task.from_separate_data(
        model=task.model, 
        trusted_data=dataset_to_dict(task.trusted_data),
        clean_test_data=dataset_to_dict(task.test_data.normal_data), 
        anomalous_test_data=dataset_to_dict(task.test_data.anomalous_data)
    )
