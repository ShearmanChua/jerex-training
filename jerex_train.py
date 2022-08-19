import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configs import TrainConfig
from jerex import model, util

from clearml import Dataset

cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)


@hydra.main(config_name='train', config_path='configs/docred_joint')
def train(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    util.config_to_abs_paths(cfg.datasets, 'train_path', 'valid_path', 'test_path', 'types_path')
    util.config_to_abs_paths(cfg.model, 'tokenizer_path', 'encoder_path')
    util.config_to_abs_paths(cfg.misc, 'cache_path')

    from clearml import Task
    Task.add_requirements("hydra-core")
    task = Task.init(project_name='Jerex_DWIE', task_name='train Jerex')
    task.set_base_docker("nvcr.io/nvidia/pytorch:20.12-py3")
    task.connect(cfg)

    task.execute_remotely()

    clearml_train_path = get_clearml_file_path(cfg.clearml.dataset_project,cfg.clearml.dataset_name,cfg.clearml.train_file_name)
    util.config_clearml_paths(cfg.datasets, 'train_path', clearml_train_path)
    clearml_valid_path = get_clearml_file_path(cfg.clearml.dataset_project,cfg.clearml.dataset_name,cfg.clearml.valid_file_name)
    util.config_clearml_paths(cfg.datasets, 'valid_path', clearml_valid_path)
    clearml_types_path = get_clearml_file_path(cfg.clearml.dataset_project,cfg.clearml.dataset_name,cfg.clearml.types_file_name)
    util.config_clearml_paths(cfg.datasets, 'types_path', clearml_types_path)

    model.train(cfg)

def get_clearml_file_path(dataset_project,dataset_name,file_name):

    print("Getting files from: ",dataset_project,dataset_name,file_name)

    # get uploaded dataset from clearML
    dataset_dict = Dataset.list_datasets(
        dataset_project=dataset_project, partial_name=dataset_name, only_completed=False
    )

    datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    dataset_obj = datasets_obj[::-1][0]
    
    folder = dataset_obj.get_local_copy()

    file = [file for file in dataset_obj.list_files() if file==file_name][0]

    file_path = folder + "/" + file

    return file_path



if __name__ == '__main__':
    train()
