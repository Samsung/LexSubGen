from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from overrides import overrides
from scipy.sparse import csr_matrix
from scipy.special import softmax

from lexsubgen.prob_estimators.base_estimator import BaseProbEstimator
from lexsubgen.utils.file import get_emb_matrix, download_embeddings
from lexsubgen.utils.register import CACHE_DIR

WORD_EMBEDDINGS_URL = (
    "http://u.cs.biu.ac.il/~nlp/downloads/lexsub/lexsub_word_embeddings.gz"
)


class OOCProbEstimator(BaseProbEstimator):
    def __init__(
        self,
        embeddings_file: str = CACHE_DIR / "resources/lexsub_word_embeddings",
        verbose: bool = False,
    ):
        """
        Probability estimator based on out-of-context model that uses word2vecf embeddings
        (see O. Levy and Y. Goldberg "Dependency based word embeddings."). Out-of-context
        means that this model do not use target word context only computes similarity scores
        between the target word and possible substitutes.

        Args:
            embeddings_file: path ot the embeddings file
            verbose: whether to print misc information
        """
        super().__init__(verbose=verbose)
        self.logger.info("Downloading embedding matrix...")
        embeddings_file_path = Path(embeddings_file)
        if not embeddings_file_path.exists():
            download_embeddings(
                url=WORD_EMBEDDINGS_URL, dest=embeddings_file_path.parent
            )

        self.embedding_matrix, self.word2id, self.vocab = get_emb_matrix(
            embeddings_file
        )
        self.vocab_size, self.embeddings_size = self.embedding_matrix.shape
        self.logger.info("Embedding matrix download has finished.")

        # Normalize embedding vectors
        for i in range(self.embedding_matrix.shape[0]):
            self.embedding_matrix[i, :] /= np.sqrt(
                np.sum(self.embedding_matrix[i, :] ** 2)
            )

        self.name = f"#OOCSubstEstimator"
        self.descr = {
            "Prob_generator": {
                "name": "ooc",
                "vect_class": self.__class__.__name__,
                "embeddings_file": embeddings_file,
            },
        }
        self.vocab = list(self.word2id.keys())

    @overrides
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        Process all input data with batches.

        Args:
            tokens_lists: list of tokenized sequences,  each list corresponds to one tokenized example.
            target_ids: indices of target words from all tokens lists.
                E.g.:
                token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                This means that we want to get probability distribution for words "world" and "stackoverflow".
        Returns:
            `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.
        """
        num_elements = len(tokens_lists)
        row, col, data = [], [], []
        for i, token_sequence in enumerate(tokens_lists):
            pos = target_ids[i]
            if token_sequence[pos] in self.word2id.keys():
                row.append(self.word2id[token_sequence[pos]])
                col.append(i)
                data.append(1.0)

        bag_of_words = csr_matrix(
            (data, (row, col)), shape=(self.vocab_size, num_elements)
        )
        scores = self.embedding_matrix.dot(self.embedding_matrix.T * bag_of_words)
        norm_scores = np.transpose(softmax(scores))
        return norm_scores, self.word2id
