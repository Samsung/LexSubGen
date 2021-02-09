from collections import defaultdict
from typing import List, Dict, Optional, Tuple, NoReturn

import numpy as np
from overrides import overrides

from lexsubgen.post_processors.base_postprocessor import PostProcessor
from lexsubgen.utils.lemmatize import lemmatize_batch


class RobertaPostProcessor(PostProcessor):
    def __init__(self, strategy: str = "drop_subwords"):
        super().__init__()
        self.strategy = strategy
        self.merged_vocab = {}
        self.prev_word2id = {}

        if strategy != "drop_subwords":
            raise ValueError(f"Unknown strategy: {strategy}")

    @overrides
    def transform(
        self,
        log_probs: np.ndarray,
        word2id: Dict[str, int],
        target_words: Optional[List[str]] = None,
        target_pos: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Transforms prob estimator predictions and vocabulary leaving
        only lemmas of the words from vocabulary, predictions for words
        with the same lemma are aggregated according to the chosen strategy.

        Args:
            log_probs: predicted log-probabilities for words
            word2id: vocabulary
            target_words: list of target words
            target_pos: list of target part of speech tags (optional)

        Returns:
            transformed predictions and transformed word2id
        """
        if word2id != self.prev_word2id:
            self.update_vocab(word2id)
            self.prev_word2id = word2id

        if self.strategy == "drop_subwords":
            subword_ids, subword2id = [], {}
            for i, (subword, idx) in enumerate(self.notsubword2old_id.items()):
                subword_ids.append(idx)
                subword2id[subword] = i
            return log_probs[:, subword_ids], subword2id

        raise ValueError(f"Unknown strategy: {self.strategy}")

    def update_vocab(
        self, word2id: Dict[str, int]
    ) -> NoReturn:
        self.wordform2ids = defaultdict(list)
        self.notsubword2old_id = {}
        for word, idx in word2id.items():
            if word.startswith('ġ') or word.startswith('Ġ'):
                self.wordform2ids[word[1:]].append(idx)
                self.notsubword2old_id[word[1:]] = idx
            else:
                self.wordform2ids[word].append(idx)
        self.new_word2id = dict(zip(
            self.wordform2ids.keys(),
            range(len(self.wordform2ids))
        ))
        self.new_id2word = {
            idx: word for word, idx in self.new_word2id.items()
        }
