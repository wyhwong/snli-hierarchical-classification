## Installation

Dependencies can be installed using uv:
```bash
uv sync --all-groups
```

> [!WARNING]
> Although the dependencies are installed, one may still handle CUDA libraries in OS level.

One can also directly install the dependencies using pip:
```bash
pip install -r requirements.txt
```

## Training

```bash
uv run python scripts/train_models.py
```

This script trains 5 models in sequence under `model_registry/`:
- `baseline_model` (3-way SNLI classifier)
- `stage_one_model` (binary stage-1 classifier)
- `stage_one_with_focus_learning_model` (stage-1 fine-tuned with focus learning)
- `stage_two_model` (binary stage-2 classifier)
- `stage_two_with_focus_learning_model` (stage-2 fine-tuned with focus learning)

If `model_registry/snli_roberta_0_6_data_map_coordinates.jsonl` is missing,
it is downloaded automatically before training.

## Evaluation

```bash
uv run python scripts/evaluate_models.py
```

This script evaluates:
- `baseline_model`
- a two-stage hierarchical model (`stage_one_model` + `stage_two_model`)
- a two-stage hierarchical model with focus learning
	(`stage_one_with_focus_learning_model` + `stage_two_with_focus_learning_model`)

By default, evaluation runs on:
- `snli` (`validation` split)
- `../data/snli_subset.json` (`train` split)
- `../data/snli_contrast_subset.json` (`train` split)

Outputs are written under `model_evaluation/`.

## Reproducibility

We note that the results are prepared during Fall 2025,
some OS-level dependency versions are not recorded.
Therefore the results may not be fully reproducible by other users.
However, the code and data are provided for reference and further research.
