import csv
import os
import re
import subprocess
import tempfile
from itertools import groupby
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, NoReturn, Any, Tuple

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

from lexsubgen.utils.wsi import (
    download_semeval_2013_data_if_not_exists,
    download_semeval_2010_data_if_not_exists,
)

MATCH_SEMEVAL_SCORES_RE = re.compile(r"(\w+|\w+\.\w+)(\t*-?\d+\.?\d*)+")
MATCH_TOTAL_VALUE = re.compile(r"Total (.+):(.+)")
METRICS = [
    "ARI",
    "NMI",
    "goldInstance",
    "sysInstance",
    "goldClusterNum",
    "sysClusterNum",
]
SEMEVAL_METRICS = [
    "S13_Precision",
    "S13_Recall",
    "S13_F1",
    "S13_FNMI",
    "S13_AVG",
    "S10_FScore",
    "S10_Precision",
    "S10_Recall",
    "S10_VMeasure",
    "S10_Homogeneity",
    "S10_Completeness",
    "S10_AVG",
]
ALL_METRICS = METRICS + SEMEVAL_METRICS


def compute_wsi_metrics(
    y_true: List,
    y_pred: List,
    group_by: List[str],
    context_ids: List[str],
    y_true_file: str = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Computes clustering metrics: @METRICS

    Args:
        y_true: ground truth
        y_pred: predicted labels
        group_by: @y_true and @y_pred must be grouped by @group_by
            and METRICs must be computed for each group
        context_ids: unique indexes of instances
        y_true_file: if not None true labels will be read from @y_true_file file
    """
    semeval_2013_values = compute_semeval_2013_metrics(
        y_true, y_pred, group_by, context_ids, y_true_file
    )
    semeval_2010_values = compute_semeval_2010_metrics(
        y_true, y_pred, group_by, context_ids, y_true_file
    )

    scores_per_word = compute_scores_per_word(y_true, y_pred, group_by)
    per_word_df = pd.DataFrame(scores_per_word, columns=['word'] + METRICS)
    mean_values = (
        compute_weighted_avg(per_word_df, METRICS)
        + semeval_2013_values["all"]
        + semeval_2010_values['all']
    )
    scores_per_word.append(["word_weighted_avg"] + mean_values)
    for i, word in enumerate(per_word_df.word):
        scores_per_word[i].extend(
            semeval_2013_values[word] + semeval_2010_values[word]
        )

    all_metrics = pd.DataFrame(scores_per_word, columns=["word"] + ALL_METRICS)
    mean_metrics = OrderedDict(
        (metric, value)
        for metric, value in zip(ALL_METRICS, mean_values)
    )

    return mean_metrics, all_metrics


def compute_weighted_avg(per_word_df: pd.DataFrame, metrics: List[str]) -> List[float]:
    """
    Computing weighted average for each metric

    Args:
        per_word_df: pandas dataframe containing each metric for each word
        metrics: list of metric names

    Returns:
        List of averages
    """
    avg_values = []
    n_instances = float(per_word_df["sysInstance"].sum())
    for metric in metrics:
        if metric in ["goldClusterNum", "sysClusterNum"]:
            avg_values.append(per_word_df[metric].mean())
        elif metric in ["goldInstance", "sysInstance"]:
            avg_values.append(per_word_df[metric].sum())
        else:
            w_sum = sum(per_word_df["sysInstance"] * per_word_df[metric])
            avg_values.append(w_sum / n_instances)
    return avg_values


def compute_scores_per_word(
    y_true: List, y_pred: List, group_by: List[str],
) -> List[List]:
    """
    Computes common clustering metrics per each word.
    Also aggregates words results.

    Args:
        y_true: Iterable, ground truth labels.
        y_pred: Iterable, labels from your clusterizer.
        group_by: y_true and y_pred are grouped by these values.
            It is assumed that these values are ambiguous words.

    Returns:
        computed metrics
    """
    values_per_word = []
    data = sorted(zip(y_true, y_pred, group_by), key=lambda x: x[2])
    for word, grouped in groupby(data, lambda x: x[2]):
        # unzip data
        local_y_true, local_y_pred, _ = zip(*grouped)
        word_values = compute_clustering_metrics(
            y_true=local_y_true, y_pred=local_y_pred
        )
        values_per_word.append([word] + word_values)
    return values_per_word


def _convert_labels_to_semeval2013_file_format(
    words: List, context_ids: List, labels: List, save_path: os.PathLike
) -> NoReturn:
    for i in range(len(words)):
        if not context_ids[i].startswith(words[i]):
            context_ids[i] = f"{words[i]}.{context_ids[i]}"
    with open(save_path, "w") as fd:
        writer = csv.writer(fd, delimiter=" ")
        writer.writerows(zip(words, context_ids, labels))


def compute_semeval_2013_metrics(
    gold_labels: List,
    pred_labels: List,
    group_by: List[str],
    context_ids: List[str],
    gold_labels_path: os.PathLike = None,
) -> Dict[str, List[float]]:
    """
    TODO: Add docs!
    """
    with tempfile.TemporaryDirectory() as temp_directory:
        save_path = Path(temp_directory)
        pred_labels_path = save_path / "predicted-labels.key"
        _convert_labels_to_semeval2013_file_format(
            group_by, context_ids, pred_labels, pred_labels_path
        )

        if gold_labels_path is None:
            gold_labels_path = save_path / "gold-labels.key"
            _convert_labels_to_semeval2013_file_format(
                group_by, context_ids, gold_labels, gold_labels_path
            )

        data_path = download_semeval_2013_data_if_not_exists()
        fnmi = _compute_semeval_2013_metrics(
            Path(data_path) / "scoring" / "fuzzy-nmi.jar",
            gold_labels_path,
            pred_labels_path,
        )
        fbc = _compute_semeval_2013_metrics(
            Path(data_path) / "scoring" / "fuzzy-bcubed.jar",
            gold_labels_path,
            pred_labels_path,
        )

    return {
        word: [prec, rec, f1, fnmi[word][0], (f1 * fnmi[word][0]) ** 0.5]
        for word, (prec, rec, f1) in fbc.items()
    }


def _compute_semeval_2013_metrics(
    java_file: str,
    gold_labels_path: str,
    pred_labels_path: str
) -> Dict[str, Any]:
    scores = dict()
    output = subprocess.run(
        ["java", "-jar", java_file, gold_labels_path, pred_labels_path],
        capture_output=True
    )
    for line in output.stdout.decode().split("\n"):
        if MATCH_SEMEVAL_SCORES_RE.match(line):
            word, *values = line.split('\t')
            scores[word] = tuple(float(val)*100.0 for val in values)
    return scores


def _compute_semeval_2010_metrics(
    java_file: str,
    gold_labels_path: str,
    pred_labels_path: str
) -> Dict[str, Any]:
    scores = dict()
    output = subprocess.run(
        ["java", "-jar", java_file, gold_labels_path, pred_labels_path, "all"],
        capture_output=True
    )
    for line in output.stdout.decode().split("\n"):
        if MATCH_SEMEVAL_SCORES_RE.match(line):
            word, *values = re.split(r"\t+", line.strip())
            scores[word] = tuple(float(val)*100.0 for val in values)
        elif MATCH_TOTAL_VALUE.match(line):
            _, val = MATCH_TOTAL_VALUE.findall(line)[0]
            scores['all'] = (float(val)*100.0, -1.0*100.0, -1.0*100.0)
    return scores


def compute_semeval_2010_metrics(
    gold_labels: List,
    pred_labels: List,
    group_by: List[str],
    context_ids: List[str],
    gold_labels_path: os.PathLike = None,
) -> Dict[str, List[float]]:

    with tempfile.TemporaryDirectory() as temp_directory:
        save_path = Path(temp_directory)
        pred_labels_path = save_path / "predicted-labels.key"
        _convert_labels_to_semeval2013_file_format(
            group_by, context_ids, pred_labels, pred_labels_path
        )
        if gold_labels_path is None:
            gold_labels_path = save_path / "gold-labels.key"
            _convert_labels_to_semeval2013_file_format(
                group_by, context_ids, gold_labels, gold_labels_path
            )

        data_path = download_semeval_2010_data_if_not_exists()

        fscore = _compute_semeval_2010_metrics(
            Path(data_path) / "evaluation" / "unsup_eval" / "fscore.jar",
            gold_labels_path,
            pred_labels_path,
        )
        vmeasure = _compute_semeval_2010_metrics(
            Path(data_path) / "evaluation" / "unsup_eval" / "vmeasure.jar",
            gold_labels_path,
            pred_labels_path,
        )

    metrics = dict()
    for word, (fs, prec, rec) in fscore.items():
        vm, homogeneity, completeness = vmeasure[word]
        metrics[word] = [fs, prec, rec, vm, homogeneity, completeness, (fs * vm) ** 0.5]
    return metrics


def compute_clustering_metrics(y_true: List, y_pred: List) -> List[float]:
    """
    Computes ARI, NMI, goldInstance, sysInstance, goldClusterNum
    and sysClusterNum between predicted and gold labels

    Args:
        y_true: Iterable, ground truth labels
        y_pred: Iterable, labels from your clusterizer

    Returns:
        ARI, NMI, goldInstance, sysInstance, goldClusterNum, sysClusterNum
    """
    return [
        adjusted_rand_score(y_true, y_pred) * 100.0,
        normalized_mutual_info_score(y_true, y_pred) * 100.0,
        len(y_true),
        len(y_pred),
        len(set(y_true)),
        len(set(y_pred)),
    ]
