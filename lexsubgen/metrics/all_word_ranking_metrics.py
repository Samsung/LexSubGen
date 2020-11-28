from collections import OrderedDict
from typing import List, Dict, Tuple, Set, Union


def compute_precision_recall_f1_topk(
    gold_substitutes: List[str],
    pred_substitutes: List[str],
    topk_list: List[int] = (1, 3, 10),
) -> Dict[str, float]:
    """
    Method for computing k-metrics for each k in the input 'topk_list'.

    Args:
        gold_substitutes: Gold substitutes provided by human annotators.
        pred_substitutes: Predicted substitutes.
        topk_list: List of integer numbers for metrics.
        For example, if 'topk_list' equal to [1, 3, 5], then there will calculating the following metrics:
            ['Precion@1', 'Recall@1', 'F1-score@1',
             'Precion@3', 'Recall@3', 'F1-score@3',
             'Precion@5', 'Recall@5', 'F1-score@5']

    Returns:
        Dictionary that maps k-values in input 'topk_list' to computed Precison@k, Recall@k, F1@k metrics.
    """
    k_metrics = OrderedDict()
    golds_set = set(gold_substitutes)
    for topk in topk_list:
        if topk > len(pred_substitutes) or topk <= 0:
            raise ValueError(f"Couldn't take top {topk} from {len(pred_substitutes)} substitues")

        topk_pred_substitutes = pred_substitutes[:topk]

        true_positives = sum(1 for s in topk_pred_substitutes if s in golds_set)
        precision, recall, f1_score = _precision_recall_f1_from_tp_tpfp_tpfn(
            true_positives,
            len(topk_pred_substitutes),
            len(gold_substitutes)
        )
        k_metrics[f"prec@{topk}"] = precision
        k_metrics[f"rec@{topk}"] = recall
        k_metrics[f"f1@{topk}"] = f1_score
    return k_metrics


def compute_precision_recall_f1_vocab(
    gold_substitutes: List[str],
    vocabulary: Union[Set[str], Dict[str, int]],
) -> Tuple[float, float, float]:
    """
    Method for computing basic metrics like Precision, Recall, F1-score on all Substitute Generator vocabulary.
    Args:
        gold_substitutes: Gold substitutes provided by human annotators.
        vocabulary: Vocabulary of the used Substitute Generator.

    Returns:
        Precision, Recall, F1 Score
    """
    true_positives = sum(1 for s in set(gold_substitutes) if s in vocabulary)
    precision, recall, f1_score = _precision_recall_f1_from_tp_tpfp_tpfn(
        true_positives,
        len(vocabulary),
        len(gold_substitutes)
    )
    return precision, recall, f1_score


def _precision_recall_f1_from_tp_tpfp_tpfn(
    tp: int, tpfp: int, tpfn: int
) -> Tuple[float, float, float]:
    """
    Computing precision, recall and f1 score
    Args:
        tp: number of true positives
        tpfp: number of true positives + false positives
        tpfn: number of true positives + false negatives

    Returns:
        Precision, Recall and F1 score
    """
    precision, recall, f1_score = 0.0, 0.0, 0.0
    if tpfp:
        precision = tp / tpfp
    if tpfn:
        recall = tp / tpfn
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
