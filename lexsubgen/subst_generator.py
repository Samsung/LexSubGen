from itertools import groupby
from typing import Iterable, Optional, List, Tuple, Dict, Union

import numpy as np
from scipy.special import softmax

from lexsubgen.post_processors import PostProcessor
from lexsubgen.pre_processors import Preprocessor
from lexsubgen.prob_estimators import BaseProbEstimator
from lexsubgen.utils.params import build_from_config_path


def top_k_strategy(probs: np.ndarray, k: int) -> List[List[int]]:
    """
    Function that implements top-k strategy, i.e. chooses k substitutes with highest probabilities.

    Args:
        probs: probability distribution
        k: number of top substitutes to take

    Returns:
        list of chosen indexes
    """
    parted = np.argpartition(probs, kth=range(-k, 0), axis=-1)
    sorted_ids = parted[:, -k:][:, ::-1]
    return sorted_ids.tolist()


def top_p_strategy(_probs: np.ndarray, p: float) -> List[List[int]]:
    """
    Function that implement top-p strategy. Takes as much substitutes as to fulfill probability threshold.

    Args:
        _probs: probability distribution
        p: probability threshold

    Returns:
        list of chosen indexes
    """
    bs, vs = _probs.shape
    sorted_ids = np.argsort(_probs, axis=-1)[:, ::-1]
    sorted_probs = _probs[[[i] * vs for i in range(bs)], sorted_ids]
    cumsum_probs = np.cumsum(sorted_probs, axis=-1) - sorted_probs
    selected_ids = []
    for idx, group in groupby(np.argwhere(cumsum_probs < p).tolist(), lambda x: x[0]):
        selected_ids.append([pair[1] for pair in group])
    return selected_ids


class SubstituteGenerator:
    def __init__(
        self,
        prob_estimator: BaseProbEstimator,
        pre_processing: Optional[Iterable[Preprocessor]] = None,
        post_processing: Optional[Iterable[PostProcessor]] = None,
        substitute_handler=None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        """
        Class that handles generation of a substitutes. This process is splitted onto
        pre-processing, probability estimation and post-processing.

        Args:
            prob_estimator: probability estimator to be used for probability
                distribution acquisition over possible substitutes
            pre_processing: list of object that do pre-processing of original contexts
            post_processing: list of objects that do post-processing with the acquired probability distribution
            substitute_handler: processes predicted substitutes, it can lemmatize them or exclude target word
            top_k: how many substitutes to grab according to top-k strategy
            top_p: probability threshold in top-p strategy
        """
        assert (
            top_k is None or top_p is None
        ), "You shouldn't provide values for top_k and top_p methods simultaneously."
        self.prob_estimator = prob_estimator
        self.pre_processing = pre_processing or []
        self.post_processing = post_processing or []
        self.substitute_handler = substitute_handler
        if top_k is not None and top_k <= 0:
            raise ValueError("k in top-k strategy must be non-negative!")
        self.top_k = top_k
        if top_p is not None and 0.0 <= top_p <= 1.0:
            raise ValueError("p in top-p strategy should be a valid probability value!")
        self.top_p = top_p

    @classmethod
    def from_config(cls, config_path):
        """
        Create substitute generator from configuration.

        Args:
            config_path: path to configuration file.

        Returns:
            object of the SubstituteGenerator class.
        """
        subst_generator, _ = build_from_config_path(config_path)
        return subst_generator

    def get_log_probs(
        self,
        sentences: List[List[str]],
        target_ids: List[int],
        target_pos_tags: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        If `self.embedding_similarity` is true will return similarity scores.
        Process all input data with batches.

        Args:
            sentences: list of tokenized sequences,  each list corresponds to one tokenized example.
            target_ids: indices of target words from all tokens lists.
            target_pos_tags: list of target pos tags
                E.g.:
                sentences = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                target_pos_tags = ['n', 'n']
                This means that we want to get probability distribution for words "world" and "stackoverflow" and
                the targets are nouns
        Returns:
            `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.
        """

        for preprocessor in self.pre_processing:
            sentences, target_ids = preprocessor.transform(sentences, target_ids)
        log_probs, word2id = self.prob_estimator.get_log_probs(sentences, target_ids)
        for postprocessor in self.post_processing:
            target_words = [
                sentence[target_id] for sentence, target_id in zip(sentences, target_ids)
            ]
            log_probs, word2id = postprocessor.transform(
                log_probs,
                word2id,
                target_words=target_words,
                target_pos=target_pos_tags,
            )
        return log_probs, word2id

    def get_probs(
        self,
        sentences: List[List[str]],
        target_ids: List[int],
        target_pos: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Computes probability distribution over vocabulary for a given instances.

        Args:
            sentences: list of contexts.
            target_ids: list of target word ids.
            target_pos: target word pos tags
                E.g.:
                token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                This means that we want to get probability distribution for words "world" and "stackoverflow".
        Returns:
            Probability distribution over vocabulary and the relative vocabulary.
        """
        log_probs, word2id = self.get_log_probs(sentences, target_ids, target_pos)
        probs = softmax(log_probs, axis=-1)
        return probs, word2id

    def substitutes_from_probs(
        self,
        probs: np.ndarray,
        word2id: Dict,
        sentences: List[List[str]] = None,
        target_ids: List[int] = None,
        target_pos: Optional[List[str]] = None,
        target_lemmas: Optional[List[str]] = None
    ):
        id2word = {idx: word for word, idx in word2id.items()}
        if self.top_k is not None:
            selected_ids = top_k_strategy(probs, self.top_k)
        elif self.top_p is not None:
            selected_ids = top_p_strategy(probs, self.top_p)
        else:
            selected_ids = np.argsort(probs)[::-1]
        substitutes = [[id2word[idx] for idx in ids] for ids in selected_ids]

        if self.substitute_handler is not None:
            if sentences is None or target_ids is None:
                raise ValueError("Sentences and target indexes must be non None to use substitute handler")
            target_words = [
                sentence[target_id]
                for sentence, target_id in zip(sentences, target_ids)
            ]
            substitutes = self.substitute_handler.transform(
                substitutes, word2id, target_words=target_words,
                target_pos=target_pos, target_lemmas=target_lemmas
            )

        return substitutes

    @staticmethod
    def candidates_from_probs(
        probs: np.ndarray,
        word2id: Dict[str, int],
        candidates: List[List[str]],
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Ranking candidates using probability distributions over vocabulary

        Args:
            probs: probability distributions over vocabulary
            word2id: mapping from word to its index in the vocabulary
            candidates: lists of candidates to be ranked

        Returns:
            ranked_candidates_in_vocab: Ranked @candidates that are in vocabulary
            ranked_candidates: Ranked @candidates
        """

        ranked_candidates_in_vocab, ranked_candidates = [], []
        for i in range(probs.shape[0]):
            candidates_in_vocab = [w for w in candidates[i] if w in word2id]
            candidate_scores = np.array([
                probs[i, word2id[cand]] for cand in candidates_in_vocab
            ])
            candidate2rank = {
                word: (candidate_scores > score).sum() + 1
                for word, score in zip(candidates_in_vocab, candidate_scores)
            }
            candidate2rank = sorted(candidate2rank.items(), key=lambda x: x[1])

            ranked_in_vocab_local = [word for word, _ in candidate2rank]

            ranked_local = ranked_in_vocab_local.copy()
            for word in candidates[i]:
                if word not in word2id:
                    ranked_local.append(word)

            ranked_candidates_in_vocab.append(ranked_in_vocab_local)
            ranked_candidates.append(ranked_local)

        return ranked_candidates_in_vocab, ranked_candidates

    def generate_substitutes(
        self,
        sentences: List[List[str]],
        target_ids: List[int],
        target_pos: Optional[List[str]] = None,
        return_probs: bool = False,
        target_lemmas: Optional[List[str]] = None
    ) -> Union[Tuple[List[List[str]], Dict, np.ndarray], Tuple[List[List[str]], Dict]]:
        """
        Generates substitutes for a given batch of instances.

        Args:
            sentences: list of contexts
            target_ids: list of target indexes
            target_pos: list of target word pos tags
            return_probs: return substitute probabilities if True
            target_lemmas: list of target lemmas

        Returns:
            substitutes, vocabulary and optionally substitute probabilities
        """
        probs, word2id = self.get_probs(sentences, target_ids, target_pos)

        substitutes = self.substitutes_from_probs(
            probs, word2id, sentences, target_ids, target_pos, target_lemmas
        )

        # TODO: fix the following error by recomputing probabilities
        if self.substitute_handler is not None and return_probs == True:
            raise ValueError("Probabilities might be incorrect because of lemmatization in substitute_handler")

        if return_probs:
            return substitutes, word2id, probs

        return substitutes, word2id
