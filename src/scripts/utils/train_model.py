import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .datamap import select_hard_to_learn_examples
from .dataset import filter_out_invalid_labels, filter_out_neutral_labels, load_dataset
from .relabel import relabel_for_stage_one_classification, relabel_for_stage_two_classification
from .starter_code.helpers import (
    compute_accuracy,
    prepare_dataset_nli,
)


def train_model(
    model_path: str,
    data_map: pd.DataFrame,
    learning_rate: float,
    num_epochs: int,
    output_dir: str,
    dataset_id: str = "snli",
    is_focus_learning: bool = False,
    hard_to_learn_confidence_threshold: float = 0.9,
    is_stage_one_learning: bool = False,
    is_stage_two_learning: bool = False,
    task_kwargs: dict = {"num_labels": 2},
    num_preprocessing_workers: int = 2,
):

    dataset = load_dataset(dataset_id)
    dataset = filter_out_invalid_labels(dataset)

    if is_focus_learning:
        dataset["train"] = select_hard_to_learn_examples(
            dataset=dataset["train"],
            data_map=data_map,
            num_preprocessing_workers=num_preprocessing_workers,
            confidence_threshold=hard_to_learn_confidence_threshold,
        )

    if is_stage_two_learning:
        # For stage 2, remove neutral examples
        dataset = filter_out_neutral_labels(dataset)
        dataset = dataset.map(relabel_for_stage_two_classification)
    elif is_stage_one_learning:
        dataset = dataset.map(relabel_for_stage_one_classification)

    model = AutoModelForSequenceClassification.from_pretrained(model_path, **task_kwargs)
    for param in model.electra.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    prepare_train_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer, 128)

    train_dataset = dataset["train"]
    train_dataset_featurized = train_dataset.map(
        prepare_train_dataset,
        batched=True,
        num_proc=num_preprocessing_workers,
        remove_columns=train_dataset.column_names,
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
        do_train=True,
        do_eval=False,
        do_predict=False,
        per_device_train_batch_size=8,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        learning_rate=learning_rate,
    )

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
    )

    trainer.train()
    trainer.save_model()
