import datasets


def load_dataset(dataset_id: str) -> datasets.Dataset:

    if dataset_id.endswith(".json") or dataset_id.endswith(".jsonl"):
        dataset = datasets.load_dataset("json", data_files=dataset_id)
    else:
        dataset = datasets.load_dataset(dataset_id)

    return dataset


def filter_out_invalid_labels(dataset: datasets.Dataset) -> datasets.Dataset:

    dataset = dataset.filter(lambda ex: ex["label"] != -1)
    return dataset


def filter_out_neutral_labels(dataset: datasets.Dataset) -> datasets.Dataset:

    dataset = dataset.filter(lambda ex: ex["label"] != 1)
    return dataset
