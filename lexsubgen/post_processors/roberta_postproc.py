from collections import defaultdict
from typing import List, Dict, Optional, Tuple, NoReturn

import numpy as np
from overrides import overrides

from lexsubgen.post_processors.base_postprocessor import PostProcessor
from lexsubgen.utils.lemmatize import lemmatize_batch


class RobertaPostProcessor(PostProcessor):
    def __init__(self, strategy: str = "max"):
        super().__init__()
        self.strategy = strategy
        self.merged_vocab = {}
        self.prev_word2id = {}

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
            for i, (subword, idx) in enumerate(self.nonsubword2old_id.items()):
                subword_ids.append(idx)
                subword2id[subword] = i
            return log_probs[:, subword_ids], subword2id

        # Misc arrangements
        zero_column = len(word2id)
        identity_column = np.empty((log_probs.shape[0], 1))
        identity_column.fill(-1e9)
        log_probs = np.concatenate([log_probs, identity_column], axis=1)
        # probs = np.exp(log_probs)
        probs = log_probs

        # Aggregate probs for each lemma relative to pos tag
        transformed_probs = np.zeros((probs.shape[0], len(self.new_word2id)))
        parallel_form_ids = [
            self.wordform2ids[self.new_id2word[idx]]
            for idx in range(len(self.new_id2word))
        ]
        max_forms_cnt = max([len(form_ids) for form_ids in parallel_form_ids])
        # min_forms_cnt = min([len(form_ids) for form_ids in parallel_form_ids])
        parallel_form_ids = [
            form_ids + [zero_column] * (max_forms_cnt - len(form_ids))
            for form_ids in parallel_form_ids
        ]

        transformed_probs = lemmatize_batch(
            probs, parallel_form_ids, strategy=self.strategy, parallel=True
        )

        # with np.errstate(divide="ignore"):
        #     # log(0) -> inf
        #     # transformed_log_probs = res
        #     transformed_log_probs = np.log(transformed_probs)
        #     # inf -> -10^9
        #     transformed_log_probs = np.where(
        #         transformed_log_probs == -float("inf"), -1e9, transformed_log_probs
        #     )

        return transformed_probs, self.new_word2id

    def update_vocab(
        self, word2id: Dict[str, int]
    ) -> NoReturn:
        self.wordform2ids = defaultdict(list)
        self.nonsubword2old_id = {}
        for word, idx in word2id.items():
            if word.startswith('ġ') or word.startswith('Ġ'):
                self.wordform2ids[word[1:]].append(idx)
                self.nonsubword2old_id[word[1:]] = idx
            else:
                self.wordform2ids[word].append(idx)
        self.new_word2id = dict(zip(
            self.wordform2ids.keys(),
            range(len(self.wordform2ids))
        ))
        self.new_id2word = {
            idx: word for word, idx in self.new_word2id.items()
        }
