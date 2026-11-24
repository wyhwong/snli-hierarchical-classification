import json
import os

import datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .dataset import filter_out_invalid_labels, filter_out_neutral_labels, load_dataset
from .relabel import relabel_for_stage_one_classification, relabel_for_stage_two_classification
from .starter_code.helpers import (
    compute_accuracy,
    prepare_dataset_nli,
)


def evaluate_two_stage_model(
    stage_one_model_path: str,
    stage_two_model_path: str,
    output_dir: str,
    dataset_id: str = "snli",
    eval_split: str = "validation",
    task_kwargs: dict = {"num_labels": 2},
    num_preprocessing_workers: int = 2,
):

    model_map = {
        1: stage_one_model_path,
        2: stage_two_model_path,
    }
    predictions = {}

    for stage, model_path in model_map.items():

        dataset = load_dataset(dataset_id)
        dataset = filter_out_invalid_labels(dataset)

        if stage == 1:
            dataset = dataset.map(relabel_for_stage_one_classification)

        if stage == 2:
            dataset = filter_out_neutral_labels(dataset)
            dataset = dataset.map(relabel_for_stage_two_classification)

        predictions[stage] = evaluate_model(
            dataset_split=dataset[eval_split],
            model_path=model_path,
            output_dir=f"{output_dir}/stage_{stage}",
            task_kwargs=task_kwargs,
            num_preprocessing_workers=num_preprocessing_workers,
        )

    combined_preds = []
    accuracy_count = 0

    for stage_one_example in predictions[1]:

        # NOTE:
        # Stage 1 predicted_label: 0 (not neutral), 1 (neutral)
        # Stage 2 predicted_label: 0 (entailment), 1 (contradiction)

        combined_example = dict(stage_one_example)
        del combined_example["predicted_scores"]

        stage_two_example = None
        for stage_two_example in predictions[2]:
            if (stage_one_example["premise"] == stage_two_example["premise"]) and (
                stage_one_example["hypothesis"] == stage_two_example["hypothesis"]
            ):
                break

        # Ground truth label adjustment
        if (stage_two_example is not None) and (stage_one_example["label"] == 0) and (stage_two_example["label"] == 1):
            combined_example["label"] = 2  # contradiction

        # From stage 1: not neutral
        if stage_one_example["predicted_label"] == 0:
            if stage_two_example is None:
                # No matching example found in stage 2 predictions.
                continue

            if stage_two_example["predicted_label"] == 1:
                combined_example["predicted_label"] = 2  # contradiction

        if combined_example["predicted_label"] == combined_example["label"]:
            accuracy_count += 1

        combined_preds.append(combined_example)

    metrics = {"eval_accuracy": accuracy_count / len(combined_preds)}
    json.dump(metrics, open(f"{output_dir}/combined_metrics.json", "w"))


def evaluate_model(
    dataset_split: datasets.Dataset,
    model_path: str,
    output_dir: str,
    task_kwargs: dict = {"num_labels": 3},
    num_preprocessing_workers: int = 2,
):

    model = AutoModelForSequenceClassification.from_pretrained(model_path, **task_kwargs)
    for param in model.electra.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    prepare_val_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer, 128)

    dataset_featurized = dataset_split.map(
        prepare_val_dataset,
        batched=True,
        num_proc=num_preprocessing_workers,
        remove_columns=dataset_split.column_names,
    )
    compute_metrics = compute_accuracy

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=False,
        do_eval=True,
        do_predict=False,
        per_device_eval_batch_size=8,
    )

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
    )

    results = trainer.evaluate()
    # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
    #
    # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
    # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
    # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
    # values that it returns.

    os.makedirs(training_args.output_dir, exist_ok=True)

    with open(
        os.path.join(training_args.output_dir, "eval_metrics.json"),
        encoding="utf-8",
        mode="w",
    ) as f:
        json.dump(results, f)

    predictions = []

    with open(
        os.path.join(training_args.output_dir, "eval_predictions.jsonl"),
        encoding="utf-8",
        mode="w",
    ) as f:
        for i, example in enumerate(dataset_split):
            example_with_prediction = dict(example)
            example_with_prediction["predicted_scores"] = eval_predictions.predictions[i].tolist()
            example_with_prediction["predicted_label"] = int(eval_predictions.predictions[i].argmax())
            f.write(json.dumps(example_with_prediction))
            f.write("\n")
            predictions.append(example_with_prediction)

    return predictions
