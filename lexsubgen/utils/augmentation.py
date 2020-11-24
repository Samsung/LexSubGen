from typing import List, Optional

import numpy as np


def pick_random_target(
    sequence_length: int,
    proportion: float = 0.5,
    max_num: Optional[int] = None,
    strategy: str = "random",
    success_prob: float = 0.5,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Pick random target indices.
    Available picking from binomial distribution with success_prob
    or from uniform distribution.

    Args:
        sequence_length: number of elements in sequence
        proportion: Fraction of tokens in utterance to be candidates to replace
        max_num: Maximum number of picked indices. If equal to None,
        maximum number is sequence_length * proportion
        success_prob: probability to pick target if strategy equal to binomial
        strategy: "binomial" or "random"
        seed: random seed
    Returns:
        picked_ids: list of picked indices
    """
    assert proportion <= 1.0
    candidate_ids = np.arange(sequence_length)
    if max_num is None:
        max_num = 1
    samples_num = min(
        len(candidate_ids), min(max_num, int(proportion * sequence_length))
    )
    np.random.seed(seed)
    if strategy == "binomial":
        mask = np.random.binomial(n=1, p=success_prob, size=sequence_length).astype(
            bool
        )
    elif strategy == "random":
        mask = np.random.choice(candidate_ids, size=samples_num, replace=False)
    else:
        raise ValueError(
            f"Invalid strategy name: {strategy}. Available strategies is 'random' or 'binomial'."
        )
    picked_ids = candidate_ids[mask][:samples_num]
    return picked_ids


def get_slot_positions(sequences: List[List[str]]) -> List[np.ndarray]:
    """
    Return indices of B- or I- tags. Input sequences must be in BIO format!
    If there are no B/I tags indices will be equal to 0, 1, ..., sequence length.

    Args:
        sequences: tokenized sentences of slot tags in BIO format
    Returns:
        slot_ids: indices of B-/I- tags for each sequence
    """
    slot_ids = []
    for sequence in sequences:
        ids = []
        for idx, slot in enumerate(sequence):
            if slot.lower().startswith("b-") or slot.lower().startswith("i-"):
                ids.append(idx)
        if not ids:
            ids = range(len(sequence))
        slot_ids.append(np.array(ids))
    return slot_ids


class SpaceTokenizer:
    """
    Tokenizer that split sentence onto tokens on space symbols.
    It's just a wrapper around `str.split` method.
    """

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes given text on space symbols.
        Args:
            text: Input text, e.g. "Hello , world !"
        Returns:
            tokens: List of tokens
        Examples:
            >>> tokenizer = SpaceTokenizer()
            >>> text = "Hello , world !"
            >>> tokenizer.tokenize(text)
            ["Hello", ",", "world", "!"]
        """
        tokens = text.split()
        return tokens

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)
