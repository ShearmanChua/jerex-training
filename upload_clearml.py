from clearml import Dataset,Task
import json

def create_dataset(folder_path, dataset_project, dataset_name):
    parent_dataset = _get_last_child_dataset(dataset_project, dataset_name)
    if parent_dataset:
        print("create child")
        parent_dataset.finalize()
        child_dataset = Dataset.create(
            dataset_name, dataset_project, parent_datasets=[parent_dataset]
        )
        # child_dataset.add_files(folder_path)
        # ipdb.set_trace()
        child_dataset.sync_folder(folder_path)
        child_dataset.upload()
        # child_dataset.finalize()
        return child_dataset
    else:
        print("create parent")
        dataset = Dataset.create(dataset_name, dataset_project)
        # dataset.add_files(folder_path)
        dataset.sync_folder(folder_path)
        dataset.upload(output_url='s3://experiment-logging/multimodal')
        # dataset.finalize()
        return dataset


def _get_last_child_dataset(dataset_project, dataset_name):
    datasets_dict = Dataset.list_datasets(
        dataset_project=dataset_project, partial_name=dataset_name, only_completed=False
    )
    if datasets_dict:
        datasets_dict_latest = datasets_dict[-1]
        return Dataset.get(dataset_id=datasets_dict_latest["id"])

def main():

    # task = Task.init(project_name="ontonotes", task_name="delete dataset")
    # Dataset.delete(dataset_id='c8e87a4a1a804206a9d026a3470247c9')

    task = Task.init(project_name="Jerex_DWIE", task_name="upload Docred data")
    dataset = create_dataset(
        folder_path="data/datasets/docred_joint",
        dataset_project="datasets/jerex_DWIE",
        dataset_name="DOCRED data",
    )
    dataset.finalize()

if __name__ == '__main__':
    main()
