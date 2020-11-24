from typing import List, Dict, Optional, Tuple, NoReturn

import numpy as np
from overrides import overrides

from lexsubgen.post_processors.base_postprocessor import PostProcessor
from lexsubgen.utils.lemmatize import get_all_vocabs, lemmatize_batch


class Lemmatizer(PostProcessor):
    def __init__(self, lemmatizer: str, strategy: str, n_parallel_forms: int = 5):
        """
        Post-processor that squash vocabulary to lemmas vocabulary
        and aggregates probabilities of words in their lemmas according
        to strategy (commonly, max or sum).

        Args:
            lemmatizer: lemmatizer type (currently support spacy and nltk)
            strategy: log-probs aggregation strategy, method of `np.ndarray`
            n_parallel_forms: maximum number of word forms for processing in parallel
        """
        super().__init__()
        self.lemmatizer = lemmatizer
        self.strategy = strategy
        self.n_parallel_forms = n_parallel_forms
        self.merged_vocab = {}
        self.prev_word2id = {}
        self.pos_lemma2words = {}

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
        if target_pos is not None:
            unique_pos_tags = list(set(target_pos))
        else:
            unique_pos_tags = ["n"]
            target_pos = ["n"] * log_probs.shape[0]

        if word2id != self.prev_word2id:
            self.update_vocab(word2id, unique_pos_tags, reset=True)

        if not all(pos_tag in self.pos_lemma2words for pos_tag in unique_pos_tags):
            new_pos_tags = [
                pos_tag
                for pos_tag in unique_pos_tags
                if pos_tag not in self.pos_lemma2words
            ]
            self.update_vocab(word2id, new_pos_tags)

        self.prev_word2id = word2id

        # Misc arrangements
        zero_column = len(word2id)
        identity_column = np.empty((log_probs.shape[0], 1))
        identity_column.fill(-1e9)
        log_probs = np.concatenate([log_probs, identity_column], axis=1)
        probs = np.exp(log_probs)

        # Aggregate probs for each lemma relative to pos tag
        transformed_probs = np.zeros((probs.shape[0], len(self.merged_vocab)))
        pos_ids = {}
        for pos_tag in unique_pos_tags:
            ids = [idx for idx, pos in enumerate(target_pos) if pos == pos_tag]
            pos_ids[pos_tag] = ids

            forms_ids = [
                self.pos_lemma2words[pos_tag][lemma]
                if self.pos_lemma2words[pos_tag][lemma]
                else [zero_column]
                for lemma in self.merged_vocab.keys()
            ]
            parallel_forms_ids, linear_forms_ids = [], []
            parallel_forms, linear_forms = [], []
            for idx, form_ids in enumerate(forms_ids):
                if len(form_ids) <= self.n_parallel_forms:
                    parallel_forms.append(idx)
                    parallel_forms_ids.append(form_ids)
                else:
                    linear_forms.append(idx)
                    linear_forms_ids.append(form_ids)

            max_forms_cnt = max(len(x) for x in parallel_forms_ids)
            # if cols = [1, 2, 3], and max_forms_cnt = 7 then cols will be equal to [1, 2, 3, n, n, n, n],
            # where n is zero_column
            parallel_forms_ids = [
                form_ids + [zero_column] * (max_forms_cnt - len(form_ids))
                for form_ids in parallel_forms_ids
            ]
            res_split = [
                lemmatize_batch(
                    probs[ids], parallel_forms_ids, self.strategy, parallel=True
                ),
                lemmatize_batch(
                    probs[ids], linear_forms_ids, self.strategy, parallel=False
                ),
            ]
            res = np.zeros((len(ids), len(self.merged_vocab)))
            res[:, parallel_forms] = res_split[0]
            res[:, linear_forms] = res_split[1]

            # Non-parallel version
            # res = lemmatize_batch(probs[ids],
            #                       forms_ids,
            #                       self.strategy,
            #                       parallel=False)
            transformed_probs[ids] = res

        with np.errstate(divide="ignore"):
            # log(0) -> inf
            # transformed_log_probs = res
            transformed_log_probs = np.log(transformed_probs)
            # inf -> -10^9
            transformed_log_probs = np.where(
                transformed_log_probs == -float("inf"), -1e9, transformed_log_probs
            )

        return transformed_log_probs, self.merged_vocab

    def update_vocab(
        self, word2id: Dict[str, int], pos_tags: List[str], reset: bool = False
    ) -> NoReturn:
        """
        Updates merged vocabulary and pos dependent lemma to word forms mapping.

        Args:
            word2id: vocabulary as a mapping from words to indexes
            pos_tags: list of part-of-speech tags
            reset: whether to reset merged vocab and pos dependent
            lemma to word forms mapping to default values before update

        """
        if reset:
            self.merged_vocab = {}
            self.pos_lemma2words = {}
        # Get vocabs for all pos tags
        vocabs = {}
        for pos_tag in pos_tags:
            lemma2words, transformed_word2id = get_all_vocabs(
                word2id, self.lemmatizer, pos_tag
            )
            vocabs[pos_tag] = transformed_word2id
            self.pos_lemma2words[pos_tag] = lemma2words

        # Merge vocabularies
        for pos_tag in pos_tags:
            for lemma in vocabs[pos_tag]:
                if lemma not in self.merged_vocab:
                    self.merged_vocab[lemma] = len(self.merged_vocab)
                    # add lemma to word ids mapping
                    for pos_tag_ in self.pos_lemma2words.keys():
                        if lemma not in self.pos_lemma2words[pos_tag_]:
                            self.pos_lemma2words[pos_tag_][lemma] = []
