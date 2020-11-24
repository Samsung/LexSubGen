import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from scipy.special import softmax


class BaseProbEstimator:
    def __init__(self, verbose: bool = False):
        """
        Abstract class that defines basic methods for probability estimators.

        Args:
            verbose: whether to print misc information
        """
        self.verbose = verbose
        self.logger = logging.getLogger(Path(__file__).name)
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        self.output_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.output_handler.setFormatter(formatter)
        self.logger.addHandler(self.output_handler)

    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        Process all input data with batches.

        Args:
            tokens_lists: list of tokenized sequences,  each list corresponds to one tokenized example.
            target_ids: indices of target words from all tokens lists.
        Returns:
            `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.

        Examples:
            >>> token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
            >>> target_ids_list = [1,2]
            >>> self.get_log_probs(tokens_lists, target_ids)
            # This means that we want to get probability distribution for words "world" and "stackoverflow".
        """
        raise NotImplementedError()

    def get_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Computes probability distribution over vocabulary for a given instances.

        Args:
            tokens_lists: list of contexts.
            target_ids: list of target word ids.
                E.g.:
                token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                This means that we want to get probability distribution for words "world" and "stackoverflow".
        Returns:
            Probability distribution over vocabulary and the relative vocabulary.
        """
        logits, word2id = self.get_log_probs(tokens_lists, target_ids)
        probs = softmax(logits, axis=-1)
        return probs, word2id
