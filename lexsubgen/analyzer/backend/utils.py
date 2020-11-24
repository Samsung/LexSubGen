import errno
import re
import socket
import traceback
from typing import List, Dict

import numpy as np
from flask import jsonify

from lexsubgen.metrics.candidate_ranking_metrics import compute_gap, WORD_PAT_RE
from lexsubgen.utils.wordnet_relation import get_wordnet_relation


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        addr, port = sock.getsockname()
    return port


def check_and_assign(port: int, verbose: bool = False):
    """
    Function that check given port for starting server
    :param port: number of port
    :param verbose:
    :return: given port if it's free or number of new free port
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
    except OSError as e:
        if e.errno == errno.EADDRINUSE and verbose:
            print(f" * Port {port} already in use!")
        port = get_free_port()
        if verbose:
            print(f" * Running server on port {port}")
    sock.close()
    return port


class EndpointAction:
    def __init__(self, _action, _action_name, _server_name):
        self.action = _action
        self.action_name = _action_name
        self.server_name = _server_name

    def __call__(self):
        try:
            response = self.action()
        except Exception as e:
            full_traceback = traceback.format_exc()
            return jsonify(
                status=500,
                err_msg=str(e),
                traceback=full_traceback,
                name=self.server_name,
            )
        if "index" in response.keys():
            return response["index"]
        return jsonify(**response, status=200, name=self.server_name)


class SubstResponse:
    def __init__(
        self,
        substitutes: List[Dict[str, List[str]]],
        target_words: List[str],
        scores: np.ndarray,
        pos_tags: List[str],
        word2id: Dict[str, int],
    ):
        self.substitutes = substitutes
        self.target_words = target_words
        self.pos_tags = pos_tags or ["n"] * len(self.target_words)
        self.scores = scores
        self.word2id = word2id

    def make_word_info(
        self,
        word: str,
        target_word: str,
        target_pos_tag: str,
        scores: np.ndarray,
        gold_subst: List[str] = None,
    ) -> Dict:
        word_score, word_rank = 0.0, -1
        if word in self.word2id.keys():
            word_score = scores[self.word2id[word]]
            word_rank = (scores > word_score).sum() + 1

        is_true_positive = False
        if gold_subst and word in gold_subst:
            is_true_positive = True
        return {
            "word": word,
            "score": int(round(word_score * 100, 2)),
            "rank": int(word_rank),
            "tp": is_true_positive,
            "wordnet_relation": get_wordnet_relation(target_word, word, target_pos_tag),
        }

    def keys(self):
        return [str(i) for i in range(len(self.substitutes))]

    @staticmethod
    def rank_comporator(item) -> int:
        rank = item["rank"]
        return rank if rank > 0 else 1e9

    def __getitem__(self, item):
        if item not in self.keys():
            raise ValueError(f"Invalid key for SubstResponse object: {item}")
        idx = int(item)
        target_word = self.target_words[idx]
        pos_tag = self.pos_tags[idx]
        scores = self.scores[idx, :]
        generated_substs = self.substitutes[idx]["generated"]
        gold_substs = self.substitutes[idx]["gold"]
        candidates = self.substitutes[idx]["candidates"]
        gold_weights = self.substitutes[idx]["gold_weights"]
        recall_at_ten, gap = None, None
        ranked_candidates = None
        if gold_substs:
            recall_at_ten = len(set(gold_substs) & set(generated_substs)) / len(
                gold_substs
            )
            recall_at_ten = round(recall_at_ten * 100, 2)
            if candidates and gold_weights:
                candidates = [w for w in candidates if WORD_PAT_RE.match(w)]
                gold_map = {
                    word: weight
                    for word, weight in zip(gold_substs, gold_weights)
                    if WORD_PAT_RE.match(word)
                }
                gap, ranked_candidates = compute_gap(
                    gold_mapping=gold_map,
                    candidates=candidates,
                    model_prediction=scores,
                    word2id=self.word2id,
                    return_ranked_candidates=True,
                )
                gap = round(gap * 100, 2)
        response = {
            "generated_substitutes": [
                self.make_word_info(w, target_word, pos_tag, scores, gold_substs)
                for i, w in enumerate(generated_substs)
            ],
            "gold_substitutes": [
                self.make_word_info(w, target_word, pos_tag, scores)
                for w in gold_substs
            ]
            if gold_substs
            else None,
            "target_word": self.make_word_info(
                target_word, target_word, pos_tag, scores
            ),
            "recall_at_ten": recall_at_ten,
            "gap": gap,
            "ranked_candidates": [
                self.make_word_info(w, target_word, pos_tag, scores, gold_substs)
                for w in ranked_candidates
            ]
            if ranked_candidates
            else None,
        }

        if response["gold_substitutes"]:
            response["gold_substitutes"].sort(key=self.rank_comporator)

        return response


class EvalResponse:
    def __init__(self, metrics: Dict[str, float]):
        self.metrics = self.process_metric_names(metrics)

    @staticmethod
    def process_metric_names(metrics: Dict[str, float]) -> Dict[str, float]:
        processed_metrics = dict()
        for name, value in metrics.items():
            processed_name = name.title()
            processed_name = processed_name.replace("_", " ")
            processed_metrics[processed_name] = value
        return processed_metrics

    def keys(self):
        return self.metrics.keys()

    def __getitem__(self, item):
        if item not in self.keys():
            raise ValueError(f"Invalid key for EvalResponse object: {item}")
        return self.metrics[item]


class DatasetLoadResponse:
    def __init__(self, samples: List[Dict]):
        self.dataset = samples

    def keys(self):
        return [str(i) for i in range(len(self.dataset))]

    def __getitem__(self, item):
        if item not in self.keys():
            raise ValueError(f"Invalid key for SubstResponse object: {item}")
        idx = int(item)
        sample = self.dataset[idx]
        context = sample["context"]
        target_word = re.findall("@\w+@", context)[0][1:-1]
        pos_tag = sample.get("target_pos", "n").lower()
        gold_wordnet_relations = None
        gold_substitutes = sample.get("gold", None)
        if gold_substitutes:
            gold_substitutes = [word.strip() for word in gold_substitutes.split(",")]
            gold_wordnet_relations = [
                get_wordnet_relation(target_word, gold, pos_tag)
                for gold in gold_substitutes
            ]
        candidates = sample.get("candidates", None)
        if candidates:
            candidates = [word.strip() for word in candidates.split(",")]
        annotations = sample.get("annotations", None)
        if annotations:
            annotations = [int(n.strip()) for n in annotations.split(",")]
        return {
            "context": context,
            "gold_substitutes": gold_substitutes,
            "gold_wordnet_relations": gold_wordnet_relations,
            "candidates": candidates,
            "annotations": annotations,
            "pos_tag": pos_tag,
        }
