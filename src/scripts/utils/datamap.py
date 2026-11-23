import os
from urllib.request import urlretrieve

import datasets
import pandas as pd


DATA_MAP_URL = "https://raw.githubusercontent.com/allenai/cartography/main/data/data_map_coordinates/snli_roberta_0_6_data_map_coordinates.jsonl"


def download_data_mapping(output_path: str) -> None:

    base_dir = os.path.dirname(output_path)
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Creating it.")
        os.makedirs(base_dir)

    urlretrieve(DATA_MAP_URL, output_path)
    print(f"Downloaded data map to {output_path}")


def load_data_mapping(file_path: str) -> pd.DataFrame:
    """Load the data mapping from a JSONL file into a pandas DataFrame."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please download it first.")

    data_map = pd.read_json(file_path, lines=True)
    return data_map


def select_hard_to_learn_examples(
    dataset: datasets.Dataset,
    data_map: pd.DataFrame,
    num_preprocessing_workers: int = 2,
    confidence_threshold: float = 0.9,
) -> datasets.Dataset:

    dataset = dataset.map(
        lambda ex, idx: {
            "confidence": data_map.loc[idx, "confidence"],
            "variability": data_map.loc[idx, "variability"],
        },
        with_indices=True,
        num_proc=num_preprocessing_workers,
    )
    dataset = dataset.filter(lambda ex: ex["confidence"] < confidence_threshold)
    dataset = dataset.remove_columns(["confidence", "variability"])

    return dataset
