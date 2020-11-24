from typing import List, Dict, Union, Tuple, Optional

import numpy as np


def precision_recall_f1_score(
    gold_substitutes: List[str],
    model_prediction: np.ndarray,
    word2id: Dict[str, int],
    k_list: Optional[List[int]] = None,
) -> Tuple[float, float, float, Dict[int, Tuple[float, float, float]]]:
    """
    Method for computing basic metrics like Precision, Recall, F1-score on all Substitute Generator vocabulary.
    Also this method computes k-metrics for each k in input 'k_list'.

    Args:
        gold_substitutes: List of gold substitutes.
        model_prediction: NumPy array with probabilities of generated substitutes.
        word2id: Dictionary that maps word from Substitute Generator vocabulary to its index.
        k_list: List of integer numbers for metrics.
        For example, if 'k_list' equal to [1, 3, 5], then there will calculating the following metrics:
            ['Precion@1', 'Recall@1', 'F1-score@1',
             'Precion@3', 'Recall@3', 'F1-score@3',
             'Precion@5', 'Recall@5', 'F1-score@5']

    Returns:
        precision: Precision score on all vocabulary.
        recall: Recall score on all vocabulary.
        f1_score: F1-score on all vocabulary.
        k_metrics: Dictionary that maps k-values in input 'k_list' to computed Precison@k, Recall@k, F1@k metrics.
    """
    number_of_golds = len(gold_substitutes)
    column_ids = model_prediction.nonzero()[0]

    # All values in k_list must be less or equal than number of predicted substitutes
    if k_list is not None:
        k_list = [k for k in k_list if k <= len(column_ids)]
    sorted_ids = model_prediction.argsort()[::-1]
    gold_ids = [word2id[word] for word in gold_substitutes if word in word2id]
    precision, recall, f1_score = get_precision_recall_f1_score(
        gold_ids, sorted_ids, number_of_golds
    )

    k_metrics = dict()
    if k_list:
        for k in k_list:
            k_metrics[k] = get_precision_recall_f1_score(
                gold_ids, sorted_ids[:k], number_of_golds
            )
    return precision, recall, f1_score, k_metrics


def get_precision_recall_f1_score(
    gold_ids: Union[List[int], np.ndarray],
    predicted_ids: Union[List[int], np.ndarray],
    number_of_golds: int = None,
    k: int = None,
) -> Tuple[float, float, float]:
    """
    Method for computing following metrics:
        1. Precision
        2. Recall
        3. F1-score

    Args:
        gold_ids: Indices of gold substitutes in Substitute Generator vocabulary.
        predicted_ids: Indices of predicted substitutes in Substitute Generator vocabulary.
        number_of_golds: Number of gold substitutes.
        k: Integer number of truncating a maximum number of substitutes to k.
    Returns:
        precision: Computed Precision metric.
        recall: Computed Recall metric.
        f1_score: Computed F1-score metric.
    """
    if k is not None:
        predicted_ids = predicted_ids[:k]
    number_of_true_positive = np.in1d(predicted_ids, gold_ids).sum()
    if number_of_golds is None:
        number_of_golds = len(gold_ids)

    precision, recall, f1_score = 0.0, 0.0, 0.0
    if len(predicted_ids):
        precision = number_of_true_positive / len(predicted_ids)
    if number_of_golds:
        recall = number_of_true_positive / number_of_golds
    if precision and recall:
        f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def oot_score(golds: Dict[str, int], substitutes: List[str]):
    """
    Method for computing Out-Of-Ten score

    Args:
        golds: Dictionary that maps gold word to its annotators number.
        substitutes: List of generated substitutes.
    Returns:
        score: Computed OOT score.
    """
    score = 0
    for subst in substitutes:
        if subst in golds:
            score += golds[subst]
    score = score / sum([value for value in golds.values()])
    return score
