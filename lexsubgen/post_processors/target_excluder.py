from typing import Dict, List, Optional, Tuple

import numpy as np
from overrides import overrides

from lexsubgen.post_processors.base_postprocessor import PostProcessor
from lexsubgen.utils.lemmatize import lemmatize_words, get_all_vocabs

DEBUG_FILE = 'debug/mt5.txt'


class TargetExcluder(PostProcessor):
    def __init__(self, lemmatizer: Optional[str] = None, use_pos_tag: bool = True, debug: bool = False):
        """
        PostProcessor that excludes target word forms from the prediction.

        Args:
            lemmatizer: lemmatizer to use (currently support nltk and spacy lemmatizers)
        """
        super(TargetExcluder, self).__init__()
        self.lemmatizer = lemmatizer
        self.use_pos_tag = use_pos_tag
        self.debug = debug
        self.pos_lemma2words = {}
        # In the case of multi-subword generation word2id could change
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
        Abstract method that transforms prob estimator predictions.

        Args:
            log_probs: predicted log-probabilities for words
            word2id: vocabulary
            target_words: list of target words
            target_pos: list of target part of speech tags (optional)

        Returns:
            transformed predictions and transformed word2id
        """
        if self.debug:
            with open(DEBUG_FILE, 'a') as file_debug:
                id2w = {v: k for (k, v) in word2id.items()}
                for i in range(log_probs.shape[0]):
                    ids = np.where(log_probs[i])[0]
                    words = [f'"{id2w[id_]}" {log_prob:.3f}' for id_, log_prob in zip(ids, log_probs[i][ids])]
                    print(' '.join(words), file=file_debug, flush=True)

        if target_pos is not None and self.use_pos_tag:
            unique_pos_tags = list(set(target_pos))
        else:
            unique_pos_tags = ["n"]
            target_pos = ["n"] * log_probs.shape[0]

        if word2id != self.prev_word2id:
            self.update_pos_lemma2words(word2id, unique_pos_tags, reset=True)

        if not all([pos_tag in self.pos_lemma2words for pos_tag in unique_pos_tags]):
            self.update_pos_lemma2words(
                word2id,
                [
                    pos_tag
                    for pos_tag in unique_pos_tags
                    if pos_tag not in self.pos_lemma2words
                ],
            )

        self.prev_word2id = word2id

        target_lemmas = lemmatize_words(
            target_words, self.lemmatizer, target_pos, verbose=False
        )

        for i in range(log_probs.shape[0]):
            target_lemma, pos_tag = target_lemmas[i], target_pos[i]
            if target_lemma not in self.pos_lemma2words[pos_tag]:
                continue
            exclude_indexes = self.pos_lemma2words[pos_tag][target_lemma]
            log_probs[i, exclude_indexes] = -1e9

        if self.debug:
            with open(DEBUG_FILE, 'a') as file_debug:
                id2w = {v: k for (k, v) in word2id.items()}
                for i in range(log_probs.shape[0]):
                    ids = np.where(log_probs[i])[0]
                    words = [f'"{id2w[id_]}" {log_prob:.3f}' for id_, log_prob in zip(ids, log_probs[i][ids])]
                    print(' '.join(words), file=file_debug, flush=True)

        return log_probs, word2id

    def update_pos_lemma2words(
            self, word2id: Dict[str, int], pos_tags: List[str], reset: bool = False
    ) -> None:
        """Updates pos dependent lemma to word forms mapping.

        Args:
            word2id: vocabulary as a mapping from words to indexes
            pos_tags: list of part-of-speech tags
            reset: whether to pos dependent lemma to word forms
            mapping to default values before update

        """
        if reset:
            self.pos_lemma2words = {}
        for pos_tag in pos_tags:
            lemma2words, _ = get_all_vocabs(
                word2id, self.lemmatizer, pos_tag, verbose=True
            )
            self.pos_lemma2words[pos_tag] = lemma2words
