import gc
from collections import defaultdict
from typing import List, Dict

import numpy as np
from scipy.spatial.distance import cdist
from torch.cuda import empty_cache

from lexsubgen.prob_estimators import BaseProbEstimator

SIMILARITY_FUNCTIONS = ("dot-product", "cosine", "euclidean")


class EmbSimProbEstimator(BaseProbEstimator):
    loaded = defaultdict(dict)

    def __init__(
        self,
        model_name: str,
        verbose: bool = False,
        sim_func: str = "dot-product",
        temperature: float = 1.0,
    ):
        """
        Class that provides an ability to acquire substitutes distribution
        according to the embedding similarity of the target word and a substitute.

        Args:
            model_name: name of the underlying vectorization model.
            verbose: verbosity level, if its true would print some misc info.
            sim_func: name of the method to use in order to compute similarity score.
            temperature: temperature that should be applied to the output logits.
        """
        super(EmbSimProbEstimator, self).__init__(verbose=verbose)
        self.model_name = model_name
        self.temperature = temperature
        if sim_func not in SIMILARITY_FUNCTIONS:
            raise ValueError(
                f"Wrong name of the similarity function. Choose one from {SIMILARITY_FUNCTIONS}."
            )
        self.sim_func = sim_func

    def register_model(self):
        """
        Method that adds model to the memory cache if not already.
        """
        raise NotImplementedError()

    # def __del__(self):
    #     self.logger.debug(f"Deleting {self.__class__} ...")
    #     self.logger.debug(
    #         f"Object count before delete: {self.loaded[self.model_name]['ref_count']}"
    #     )
    #     self.loaded[self.model_name]["ref_count"] -= 1
    #     if self.loaded[self.model_name]["ref_count"] == 0:
    #         model_parameters = self.loaded.pop(self.model_name)
    #         for param_name, param_value in model_parameters.items():
    #             if param_name != "ref_count":
    #                 del param_value
    #         del model_parameters
    #         self.logger.debug("Object count after delete: 0")
    #     else:
    #         self.logger.debug(
    #             f"Object count after delete: {self.loaded[self.model_name]['ref_count']}"
    #         )
    #     self.logger.removeHandler(self.output_handler)
    #     # Call the garbage collector
    #     gc.collect()
    #     # Empty gpu cache
    #     empty_cache()

    def get_emb_similarity(
        self, tokens_batch: List[List[str]], target_ids_batch: List[int],
    ) -> np.ndarray:
        """
        Computes similarity between target words and substitutes
        according their embedding vectors.

        Args:
            tokens_batch: list of contexts
            target_ids_batch: list of target word ids in the given contexts

        Returns:
            similarity scores between target words and
            words from the model vocabulary.
        """
        target_words = [
            tokens[target_idx]
            for tokens, target_idx in zip(tokens_batch, target_ids_batch)
        ]

        target_word_embeddings = []
        for word in target_words:
            if word in self.word2id:
                target_word_embeddings.append(self.embeddings[self.word2id[word]])
            else:
                target_word_embeddings.append(self.get_unk_word_vector(word))
        target_word_embeddings = np.vstack(target_word_embeddings)

        if self.sim_func == "dot-product":
            logits = np.matmul(target_word_embeddings, self.embeddings.T)
        else:
            logits = 1 - cdist(
                target_word_embeddings, self.embeddings, self.sim_func
            )
        logits /= self.temperature
        return logits

    def get_unk_word_vector(self, word: str) -> np.ndarray:
        """
        This method returns vector to be used as a default if
        word is not present in the vocabulary. You may override
        this method in order to implement custom logic.

        Args:
            word: word for which the vector should be given

        Returns:
            zeros vector
        """
        # raise NotImplementedError("Override this method")
        embedding_dim = self.embeddings.shape[1]
        zeros_vector = np.zeros((1, embedding_dim))
        return zeros_vector

    @property
    def word2id(self) -> Dict[str, int]:
        """
        Attribute that acquires model vocabulary.

        Returns:
            vocabulary represented as a `dict`
        """
        return self.loaded[self.model_name]["word2id"]

    @property
    def embeddings(self) -> np.ndarray:
        """
        Attribute that acquires model word embeddings.

        Returns:
            2-D `numpy.ndarray` with rows representing word vectors.
        """
        return self.loaded[self.model_name]["embeddings"]

    @property
    def model(self):
        """
        Attribute that acquires underlying vectorization model.

        Returns:
            Vectorization model.
        """
        return self.loaded[self.model_name]["model"]
