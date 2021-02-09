import logging
from itertools import groupby
from pathlib import Path
from typing import List, Any, Optional, Dict

from tqdm import tqdm

from lexsubgen.clusterizers.agglo import SubstituteClusterizer
from lexsubgen.subst_generator import SubstituteGenerator
from lexsubgen.utils.batch_reader import BatchReader
from lexsubgen.utils.params import build_from_config_path

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)


class WSISolver:
    def __init__(
        self,
        substitute_generator: SubstituteGenerator,
        clusterizer: SubstituteClusterizer,
    ):
        """
        Class for solving Word Sense Induction problem using Substitute Generator and Clusterizer.

        Args:
            substitute_generator: object that generates possible substitutes
            clusterizer: object that clusters generated substitutes
        """
        self.substitute_generator = substitute_generator
        self.clusterizer = clusterizer

    @classmethod
    def from_config(cls, config_path: str):
        """
        Builds an object of a class that inherits from this class
            using parameters described in a config file
        Args:
            config_path: path to .jsonnet config.
                For example, see this config for LexSubEvaluation class:
                "configs/evaluations/lexsub/semeval_all_elmo.jsonnet"
        Returns: an object that was created with the parameters described in the given config
        """
        evaluation_object, _ = build_from_config_path(config_path)
        return evaluation_object

    @classmethod
    def from_configs(cls, substitute_generator_config: str, clusterizer_config: str):

        substitute_generator = SubstituteGenerator.from_config(
            substitute_generator_config
        )
        clusterizer = SubstituteClusterizer.from_config(clusterizer_config)

        return cls(substitute_generator, clusterizer)

    @staticmethod
    def optional_batch_data(data: List[Any], batch_indexes: List[int]) -> List[Any]:
        if data is None:
            return None
        return [data[idx] for idx in batch_indexes]

    def _generate_substitutes(
        self,
        tokens_lists: List[List[str]],
        target_idxs: List[int],
        batch_size: int,
        target_pos: List[str] = None,
        target_lemmas: Optional[List[str]] = None
    ) -> List[List[str]]:
        """
        Generate substitutes for a target word in a given context.
        The whole data is processed in batches.

        Args:
            tokens_lists: list of contexts represented as a sequence of tokens
            target_idxs: list of target word indexes
            batch_size: batch size
            target_lemmas: list of lemmatized target words

        Returns:
            list of generated substitutes
        """

        batch_reader = BatchReader(
            tokens_lists, target_idxs, range(len(tokens_lists)), batch_size=batch_size
        )

        substitutes = []
        for batch_data in batch_reader:
            batch_tokens, batch_target_idxs, batch_indexes = batch_data

            batch_target_lemmas = self.optional_batch_data(target_lemmas, batch_indexes)
            batch_target_pos = self.optional_batch_data(target_pos, batch_indexes)

            substs, _ = self.substitute_generator.generate_substitutes(
                batch_tokens, batch_target_idxs,
                target_pos=batch_target_pos,
                target_lemmas=batch_target_lemmas
            )
            substitutes.extend(substs)

        return substitutes

    def solve_for_word(
        self,
        tokens_lists: List[List[str]],
        target_idxs: List[int],
        target_pos: List[str] = None,
        target_lemmas: Optional[List[str]] = None,
        batch_size: int = 50
    ):
        """
        Generates substitutes for each target word.
        Vectorizes generated substitutes and clusters obtained representations.

        Args:
            tokens_lists: List of tokenized sentences.
            target_idxs: List of positions of target words for each sentence.
            target_pos: List of part of speeches.
            target_lemmas: list of lemmatized target words
            batch_size: Number of samples in batch.

        Returns:
            instance_labels: Labels of clustered instances
        """

        substitutes = self._generate_substitutes(
            tokens_lists, target_idxs, batch_size, target_pos, target_lemmas
        )

        labels = self.clusterizer.predict(substitutes)

        return labels

    def solve(
        self,
        tokens_lists: List[List[str]],
        target_idxs: List[int],
        target_pos: List[str] = None,
        group_by: Optional[List[Any]] = None,
        target_lemmas: Optional[List[str]] = None,
        batch_size: int = 50,
        verbose: bool = False
    ) -> List[Any]:
        """
        Groups @tokens_lists and @target_ids by @group_by values.
        Then generates substitutes for each grouped bunch of data.
        Vectorizes generated substitutes and clusters obtained representations.
        Combines the labels according to the initial positions of the instances.

        Args:
            tokens_lists: List of tokenized sentences.
            target_idxs: List of positions of target words for each sentence.
            target_pos: List of parts of speeches
            group_by: Groups data by these values. It might be a list of ambiguous words.
            target_lemmas: list of lemmatized target words
            batch_size: Number of samples in batch.
            verbose: Bool flag for verbosity.

        Returns:
            instance_labels: Labels of clustered instances
        """
        assert len(tokens_lists) == len(target_idxs)

        logger.info(
            f"Solving WSI task for {len(set(group_by)) if group_by is not None else 1} "
            f"ambiguous words and {len(tokens_lists)} instances ..."
        )

        if group_by is None:
            return self.solve_for_word(
                tokens_lists, target_idxs, target_pos, target_lemmas, batch_size
            )

        data = sorted(
            zip(tokens_lists, target_idxs, group_by, range(len(group_by))),
            key=lambda x: x[2]
        )

        aggregated_labels = []
        for word, local_data in tqdm(groupby(data, lambda x: x[2]),
                                     disable=not verbose,
                                     total=len(set(group_by)),
                                     desc="Solving WSI"):
            # unzip grouped data
            tokens_lists, target_idxs, _, grouped_indexes = zip(*local_data)

            grouped_target_lemmas = self.optional_batch_data(target_lemmas, grouped_indexes)
            grouped_target_pos = self.optional_batch_data(target_pos, grouped_indexes)

            labels = self.solve_for_word(
                tokens_lists, target_idxs, grouped_target_pos,
                grouped_target_lemmas, batch_size
            )

            aggregated_labels.extend(zip(grouped_indexes, labels))

        logger.info("Solving done.")

        # sort by indexes to restore original order
        aggregated_labels.sort(key=lambda x: x[0])
        return [label for _, label in aggregated_labels]

    def substitutes_generation_step(
        self,
        tokens_lists: List[List[str]],
        target_idxs: List[int],
        target_pos: List[str] = None,
        group_by: Optional[List[Any]] = None,
        target_lemmas: Optional[List[str]] = None,
        batch_size: int = 50,
        verbose: bool = False
    ) -> Dict[int, List[str]]:
        """
        Generates substitutes for each instance

        Args:
            tokens_lists: List of tokenized sentences.
            target_idxs: List of positions of target words for each sentence.
            target_pos: List of parts of speeches
            group_by: Groups data by these values. It might be a list of ambiguous words.
            target_lemmas: list of lemmatized target words.
            batch_size: Number of samples in batch.
            verbose: Bool flag for verbosity.

        Returns:
            Mapping from instance id to instance substitutes
        """
        assert len(tokens_lists) == len(target_idxs)

        if group_by is None:
            substitutes = self._generate_substitutes(
                tokens_lists, target_idxs, batch_size, target_pos, target_lemmas
            )
            return {
                idx: substs
                for idx, substs in zip(range(len(target_idxs)), substitutes)
            }

        data = sorted(
            zip(tokens_lists, target_idxs, group_by, range(len(group_by))),
            key=lambda x: x[2]
        )

        idx2substitutes = dict()
        for word, local_data in tqdm(groupby(data, lambda x: x[2]),
                                     disable=not verbose,
                                     total=len(set(group_by)),
                                     desc="Generating Substitutes"):
            # unzip grouped data
            tokens_lists, target_idxs, _, grouped_indexes = zip(*local_data)

            grouped_target_lemmas = self.optional_batch_data(target_lemmas, grouped_indexes)
            grouped_target_pos = self.optional_batch_data(target_pos, grouped_indexes)

            substitutes = self._generate_substitutes(
                tokens_lists, target_idxs, batch_size, grouped_target_pos, grouped_target_lemmas
            )

            idx2substitutes.update({
                idx: substs
                for idx, substs in zip(grouped_indexes, substitutes)
            })

        return idx2substitutes

    def clustering_step(
        self,
        idx2substitutes: Dict[int, List[str]],
        group_by: Optional[List[Any]] = None,
        verbose: bool = False,
        memory=None,
    ) -> List[Any]:
        """

        Args:
            idx2substitutes: Mapping from unique instance id to instance substitutes
            group_by: Groups data by these values. It might be a list of ambiguous words.
            verbose: Bool flag for verbosity.
            memory: Agglomerative clustering argument

        Returns:
            List of cluster labels
        """
        if group_by is None:
            substitutes = [
                idx2substitutes[idx] for idx in range(len(idx2substitutes))
            ]
            return list(self.clusterizer.predict(substitutes))

        data = sorted(zip(group_by, range(len(group_by))), key=lambda x: x[0])

        aggregated_labels = []
        for word, local_data in tqdm(
            groupby(data, lambda x: x[0]),
            disable=not verbose,
            total=len(set(group_by)),
            desc=f"Clustering Substitutes: {self.clusterizer.n_clusters}"
        ):
            # unzip grouped data
            _, indexes = zip(*local_data)
            substitutes = [idx2substitutes[idx] for idx in indexes]
            labels = self.clusterizer.predict(substitutes, memory=memory)
            aggregated_labels.extend(zip(indexes, labels))

        # sort by indexes to restore original order
        aggregated_labels.sort(key=lambda x: x[0])
        return [label for _, label in aggregated_labels]
