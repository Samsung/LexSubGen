from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


class PostProcessor:
    def __init__(self):
        """
        Abstract class for post-processing of predicted substitutes.
        One should implement transform method.
        """
        pass

    def transform(
        self,
        log_probs: np.ndarray,
        word2id: Dict[str, int],
        target_words: Optional[List[str]] = None,
        target_pos: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Abstract method that transforms prob estimator predictions
        and in some cases a vocabulary.

        Args:
            log_probs: predicted log-probabilities for words
            word2id: vocabulary
            target_words: list of target words
            target_pos: list of target part of speech tags (optional)

        Returns:
            transformed predictions and transformed word2id
        """
        raise NotImplemented


class LowercasePostProcessor(PostProcessor):
    def __init__(self, strategy: str = "sum"):
        """
        Post-processor that merges capitalized with lowercased
        form of word. Lowercased word is preserved

        Args:
            strategy: probability aggregation strategy: max or sum.
        """
        super().__init__()
        if strategy not in ("sum", "max"):
            raise ValueError("Lowercase strategy should be sum or max.")
        self.strategy = strategy
        self.transformed_word2id = None
        self.prev_word2id = {}

    def transform(
        self,
        log_probs: np.ndarray,
        word2id: Dict[str, int],
        target_words: Optional[List[str]] = None,
        target_pos: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Merge capitalized version of the words in the vocabulary
        with the lowercased one.

        Args:
            log_probs: predicted log-probabilities for words
            word2id: vocabulary
            target_words: list of target words
            target_pos: list of target part of speech tags (optional)

        Returns:
            transformed predictions and transformed word2id
        """
        if self.transformed_word2id is None or self.prev_word2id != word2id:
            self.transformed_word2id = {}
            self.merge_indices = defaultdict(list)
            for word, idx in word2id.items():
                word_lower = word.lower()
                self.merge_indices[word_lower].append(idx)
                self.transformed_word2id[word_lower] = self.transformed_word2id.get(
                    word_lower, len(self.transformed_word2id)
                )
            max_len = len(max(self.merge_indices.values(), key=lambda x: len(x)))
            identity_column_idx = log_probs.shape[1]  # Size of the original vocabulary
            self.merge_indices = [
                idxs + [identity_column_idx] * (max_len - len(idxs))
                for word, idxs in self.merge_indices.items()
            ]
        # Get indices to merge and perform this merge in order to acquire better results
        identity_column = np.empty((log_probs.shape[0], 1))
        identity_column.fill(-1e9)
        log_probs = np.concatenate([log_probs, identity_column], axis=1)
        probs = np.exp(log_probs)

        res = probs[:, self.merge_indices].__getattribute__(self.strategy)(axis=-1)

        with np.errstate(divide="ignore"):
            # log(0) -> inf
            transformed_log_probs = np.log(res)
            # inf -> -10^9
            transformed_log_probs = np.where(
                transformed_log_probs == -float("inf"), -1e9, transformed_log_probs
            )
        return transformed_log_probs, self.transformed_word2id
