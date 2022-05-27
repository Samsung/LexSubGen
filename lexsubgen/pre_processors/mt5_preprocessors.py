from typing import List, Tuple, Optional

from overrides import overrides

from lexsubgen.pre_processors.base_preprocessors import Preprocessor


class MT5Preprocessor(Preprocessor):
    """
    Preprocessor that replace target word in sentences to special token: <extra_id_0>
    """

    def __init__(self, spec_token='<extra_id_0>', verbose=False):
        """
        Preprocessor that replace target word to special token: <extra_id_0>
        """
        super().__init__()
        self.spec_token = spec_token
        self.verbose = verbose

    @overrides
    def transform(
            self, sentences: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Replace target word to special token: <extra_id_0>

        Args:
            sentences: list of sentences which are represented as a token sequence
            target_ids: indexes of target word in sentence

        Returns:
            Sentence with replaced target word
        """
        new_sentences = sentences.copy()
        for i in range(len(sentences)):
            if self.verbose:
                print(sentences[i])
            new_sentences[i][target_ids[i]] = self.spec_token
            if self.verbose:
                print(new_sentences[i])

        return new_sentences, target_ids
