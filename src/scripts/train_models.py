import os

from utils.datamap import download_data_mapping, load_data_mapping
from utils.train_model import train_model


MODEL_REGISTRY_DIR = "model_registry"
DATA_MAP_PATH = f"{MODEL_REGISTRY_DIR}/snli_roberta_0_6_data_map_coordinates.jsonl"
NUM_PREPROCESSING_WORKERS = 2


def main():
    if not os.path.exists(DATA_MAP_PATH):
        download_data_mapping(output_path=DATA_MAP_PATH)

    training_args = {
        "baseline": {
            "model_path": "google/electra-small-discriminator",
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "output_dir": f"{MODEL_REGISTRY_DIR}/baseline_model",
            "dataset_id": "snli",
            "is_focus_learning": False,
            "is_stage_one_learning": False,
            "is_stage_two_learning": False,
            "task_kwargs": {"num_labels": 3},
            "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
        },
        "stage_one": {
            "model_path": "google/electra-small-discriminator",
            "learning_rate": 5e-5,
            "num_epochs": 1,
            "output_dir": f"{MODEL_REGISTRY_DIR}/stage_one_model",
            "dataset_id": "snli",
            "is_focus_learning": False,
            "is_stage_one_learning": True,
            "is_stage_two_learning": False,
            "task_kwargs": {"num_labels": 2},
            "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
        },
        "stage_one_with_focus_learning": {
            "model_path": f"{MODEL_REGISTRY_DIR}/stage_one_model",
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "output_dir": f"{MODEL_REGISTRY_DIR}/stage_one_with_focus_learning_model",
            "dataset_id": "snli",
            "is_focus_learning": True,
            "is_stage_one_learning": True,
            "is_stage_two_learning": False,
            "task_kwargs": {"num_labels": 2},
            "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
        },
        "stage_two": {
            "model_path": "google/electra-small-discriminator",
            "learning_rate": 5e-5,
            "num_epochs": 1,
            "output_dir": f"{MODEL_REGISTRY_DIR}/stage_two_model",
            "dataset_id": "snli",
            "is_focus_learning": False,
            "is_stage_one_learning": False,
            "is_stage_two_learning": True,
            "task_kwargs": {"num_labels": 2},
            "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
        },
        "stage_two_with_focus_learning": {
            "model_path": f"{MODEL_REGISTRY_DIR}/stage_two_model",
            "learning_rate": 2e-5,
            "num_epochs": 1,
            "output_dir": f"{MODEL_REGISTRY_DIR}/stage_two_with_focus_learning_model",
            "dataset_id": "snli",
            "is_focus_learning": True,
            "is_stage_one_learning": False,
            "is_stage_two_learning": True,
            "task_kwargs": {"num_labels": 2},
            "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
        },
    }

    data_map = load_data_mapping(DATA_MAP_PATH)
    for stage, args in training_args.items():
        print(f"Training {stage} model...")
        train_model(data_map=data_map, **args)
        print(f"Finished training {stage} model.")


if __name__ == "__main__":
    main()
