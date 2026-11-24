from utils.dataset import load_dataset
from utils.evaluate_model import evaluate_model, evaluate_two_stage_model


MODEL_REGISTRY_DIR = "model_registry"
MODEL_EVALUATION_DIR = "model_evaluation"
NUM_PREPROCESSING_WORKERS = 2


def main():

    datasets = {
        "snli": {
            "dataset_id": "snli",
            "eval_split": "validation",
        },
        "snli_subset": {
            "dataset_id": "../data/snli_subset.json",
            "eval_split": "train",
        },
        "snli_contrast_subset": {
            "dataset_id": "../data/snli_contrast_subset.json",
            "eval_split": "train",
        },
    }

    for dataset_name, dataset_info in datasets.items():
        dataset_split = load_dataset(dataset_info["dataset_id"])[dataset_info["eval_split"]]
        evaluate_model(
            model_path=f"{MODEL_REGISTRY_DIR}/baseline_model",
            dataset_split=dataset_split,
            output_dir=f"{MODEL_EVALUATION_DIR}/baseline_model",
            task_kwargs={"num_labels": 3},
            num_preprocessing_workers=NUM_PREPROCESSING_WORKERS,
        )

        evaluation_args = {
            "hierarchical_model": {
                "stage_one_model_path": f"{MODEL_REGISTRY_DIR}/stage_one_model",
                "stage_two_model_path": f"{MODEL_REGISTRY_DIR}/stage_two_model",
                "output_dir": f"{MODEL_EVALUATION_DIR}/hierarchical_model/{dataset_name}",
                "dataset_id": dataset_info["dataset_id"],
                "eval_split": dataset_info["eval_split"],
                "task_kwargs": {"num_labels": 2},
                "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
            },
            "hierarchical_model_with_focus_learning": {
                "stage_one_model_path": f"{MODEL_REGISTRY_DIR}/stage_one_with_focus_learning_model",
                "stage_two_model_path": f"{MODEL_REGISTRY_DIR}/stage_two_with_focus_learning_model",
                "output_dir": f"{MODEL_EVALUATION_DIR}/hierarchical_model_with_focus_learning/{dataset_name}",
                "dataset_id": dataset_info["dataset_id"],
                "eval_split": dataset_info["eval_split"],
                "task_kwargs": {"num_labels": 2},
                "num_preprocessing_workers": NUM_PREPROCESSING_WORKERS,
            },
        }

        for model_name, args in evaluation_args.items():
            print(f"Evaluating {model_name} on dataset {args['dataset_id']}...")
            evaluate_two_stage_model(**args)
            print(f"Finished evaluating {model_name}.")


if __name__ == "__main__":
    main()
