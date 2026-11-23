def relabel_for_stage_one_classification(example: dict) -> dict:
    """
    Original labels:
    - 0: entailment
    - 1: neutral
    - 2: contradiction

    In stage 1, we relabel the samples into
    - 0: not neutral
    - 1: neutral
    """

    if example["label"] == 2:
        example["label"] = 0

    return example


def relabel_for_stage_two_classification(example: dict) -> dict:
    """
    Original labels:
    - 0: entailment
    - 1: neutral
    - 2: contradiction

    In stage 2, we relabel the samples into
    - 0: entailment
    - 1: contradiction
    """

    if example["label"] == 2:
        example["label"] = 1

    return example
