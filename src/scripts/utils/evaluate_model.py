import json
import os

import datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .relabel import relabel_for_stage_one_classification, relabel_for_stage_two_classification
from .starter_code.helpers import (
    compute_accuracy,
    prepare_dataset_nli,
)


def evaluate_model(
    model_map: dict[str, str],
    dataset_id: str = "snli",
    eval_split: str = "validation",
    task_kwargs: dict = {"num_labels": 2},
    num_preprocessing_workers: int = 2,
):
    predictions = {}

    for stage in [1, 2]:

        if dataset_id.endswith(".json") or dataset_id.endswith(".jsonl"):
            dataset = datasets.load_dataset("json", data_files=dataset_id)
        else:
            dataset = datasets.load_dataset(dataset_id)

        dataset = dataset.filter(lambda ex: ex["label"] != -1)
        if stage == 2:
            dataset = dataset.filter(lambda ex: ex["label"] != 1)

        dataset = dataset.map(
            relabel_for_stage_one_classification if stage == 1 else relabel_for_stage_two_classification
        )

        model = AutoModelForSequenceClassification.from_pretrained(model_map[stage], **task_kwargs)
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        tokenizer = AutoTokenizer.from_pretrained(model_map[stage], use_fast=True)

        prepare_val_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer, 128)

        eval_dataset = dataset[eval_split]
        eval_dataset_featurized = eval_dataset.map(
            prepare_val_dataset,
            batched=True,
            num_proc=num_preprocessing_workers,
            remove_columns=eval_dataset.column_names,
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
            output_dir=f"./evaluation/stage{stage}",
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
            eval_dataset=eval_dataset_featurized,
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

        print("Evaluation results:")
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(
            os.path.join(training_args.output_dir, "eval_metrics.json"),
            encoding="utf-8",
            mode="w",
        ) as f:
            json.dump(results, f)

        predictions[stage] = []

        with open(
            os.path.join(training_args.output_dir, "eval_predictions.jsonl"),
            encoding="utf-8",
            mode="w",
        ) as f:
            for i, example in enumerate(eval_dataset):
                example_with_prediction = dict(example)
                example_with_prediction["predicted_scores"] = eval_predictions.predictions[i].tolist()
                example_with_prediction["predicted_label"] = int(eval_predictions.predictions[i].argmax())
                f.write(json.dumps(example_with_prediction))
                f.write("\n")
                predictions[stage].append(example_with_prediction)

    combined_preds = []
    accuracy_count = 0

    for s1_example in predictions[1]:

        # NOTE:
        # Stage 1 predicted_label: 0 (not neutral), 1 (neutral)
        # Stage 2 predicted_label: 0 (entailment), 1 (contradiction)

        combined_example = dict(s1_example)
        del combined_example["predicted_scores"]

        s2_example = None
        for s2_example in predictions[2]:
            if (s1_example["premise"] == s2_example["premise"]) and (
                s1_example["hypothesis"] == s2_example["hypothesis"]
            ):
                break

        # Ground truth label adjustment
        if (s2_example is not None) and (s1_example["label"] == 0) and (s2_example["label"] == 1):
            combined_example["label"] = 2  # contradiction

        # From stage 1: not neutral
        if s1_example["predicted_label"] == 0:
            if s2_example is None:
                print("No matching example found in stage 2 predictions.")
                continue

            if s2_example["predicted_label"] == 1:
                combined_example["predicted_label"] = 2  # contradiction

        if combined_example["predicted_label"] == combined_example["label"]:
            accuracy_count += 1

        combined_preds.append(combined_example)

    print("Accuracy of combined predictions:", accuracy_count / len(combined_preds))
