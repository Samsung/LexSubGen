from typing import List, Tuple

from overrides import overrides

from lexsubgen.pre_processors.base_preprocessors import Preprocessor


def insert_pattern(
    tokens: List[str], target_idx: int, pattern: str
) -> Tuple[List[str], int]:
    """
    Pattern must contain '{predict}' token.
    Token '{target}' and token '{predict}' are replaced
        by the target word.

    Args:
        tokens: context represented as a list of tokens
        target_idx: replacement position
        pattern: string that represents a pattern, e.g. For example '{target} and {predict}'

    Returns:
        modified context and modified target position
    """
    if not pattern:
        return tokens, target_idx

    target = tokens[target_idx]

    try:
        lctx, rctx = [
            s.format(target=target).strip().split(" ")
            for s in pattern.split("{predict}")
        ]
    except ValueError:
        raise ValueError('pattern must have exactly one token "{predict}"')

    lctx, rctx = [
        [token for token in ctx if token] for ctx in [lctx, rctx]
    ]  # drop empty strings: ''

    mod_target = target_idx + len(lctx)
    mod_tokens = tokens[:target_idx] + lctx + [target] + rctx + tokens[target_idx + 1 :]

    return mod_tokens, mod_target


class PatternPreprocessor(Preprocessor):
    def __init__(
        self,
        pattern: str = "",  # '{target} (or even {predict})'
        lowercase: bool = False,
    ):
        """
        Preprocessor that applies specified pattern to a target word.

        Args:
            pattern: pattern that is applied to target word
            lowercase: whether to lowercase input sequence
        """
        super(PatternPreprocessor, self).__init__()
        assert (
            len(pattern.split("{predict}")) == 2
        ), 'pattern must have exactly one token "{predict}"'
        self.pattern = pattern
        self.lowercase = lowercase

    @overrides
    def transform(
        self, sentences: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Transforms sentence by substituting target word with a pattern, e.g. T such as Y

        Args:
            sentences: list of sentences which are represented as a token sequence
            target_ids: indexes of target word in sentence

        Returns:
            transformed sentences list and new target indexes
        """
        transformed_sentences, transformed_target_ids = [], []

        for tokens, target_id in zip(sentences, target_ids):
            tokens_, target_id_ = insert_pattern(tokens, target_id, self.pattern)
            if self.lowercase:
                tokens_ = [tok.lower() for tok in tokens_]

            transformed_sentences.append(tokens_)
            transformed_target_ids.append(target_id_)
        return transformed_sentences, transformed_target_ids
