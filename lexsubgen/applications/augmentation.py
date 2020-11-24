import re
from collections import defaultdict
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from lexsubgen.datasets.nlu import NLUDatasetReader
from lexsubgen.subst_generator import SubstituteGenerator
from lexsubgen.utils.augmentation import (
    get_slot_positions,
    pick_random_target,
    SpaceTokenizer,
)
from lexsubgen.utils.batch_reader import BatchReader


class Augmenter:
    def __init__(
        self,
        substitute_generator: SubstituteGenerator,
        max_num_augmented_utters: int = 5,
        max_substitute_usage: int = 3,
        num_replacements: int = 1,
        possibilities_proportion: float = 0.5,
        max_num_possibilities: int = None,
        tokenizer: Optional[Callable] = None,
        slot_tokenizer: Optional[Callable] = None,
        batch_size: int = 50,
        verbose: bool = False,
    ):
        """
        Class for generate augmented utterances using Substitute Generator.
        It can be used in 2 cases:
        1. Augment arbitrary utterances with method 'augment_utterances'.
        2. Augment whole dataset (like SNIPS) with method 'augment_dataset'.

        Args:
            substitute_generator: Object that generate possible substitutes
            max_num_augmented_utters: Maximum number of augmented utterances generated from one utterance
            max_substitute_usage: Maximum replacement number of substitute for one utterance
            num_replacements: Number of words which will be replaced in augmented version of utterance
            possibilities_proportion: Fraction of tokens in utterance to be candidates to replace
            max_num_possibilities: Maximum number of tokens to be candidates to replace
            tokenizer: Tokenizer for tokenizing input utterance
            slot_tokenizer: Tokenizer for tokenizing slot tags sequence
            batch_size: Number of samples in batch for substitute generator
            verbose: Bool flag for verbosity
        """

        self.substitute_generator = substitute_generator
        self.tokenizer = tokenizer or SpaceTokenizer()
        self.slot_tokenizer = slot_tokenizer or SpaceTokenizer()

        self.max_num_augmented_utters = max_num_augmented_utters
        self.max_substitute_usage = max_substitute_usage
        self.num_replacements = num_replacements
        self.possibilities_proportion = possibilities_proportion
        self.max_num_possibilities = max_num_possibilities
        self.batch_size = batch_size
        self.verbose = verbose

    def generate_possible_substitutes(
        self, utterances_tokens: List[List[str]], target_ids: List[int]
    ) -> List[List[Dict]]:
        """
        Function used to generate substitutes for given utterances on chosen indices.

        Args:
            utterances_tokens: list of tokenized utterances
            target_ids: list of chosen indices
        Returns:
            possible_substitutes: for each utterance return list of possible substitutes with their probabilities
        """
        possible_substitutes = list()
        generated = self.substitute_generator.generate_substitutes(
            utterances_tokens, target_ids, return_probs=True
        )
        utterances_substitutes, word2id, probabilities = generated
        for utterance_idx, substitutes in enumerate(utterances_substitutes):
            substitute_ids = [word2id[s] for s in substitutes]
            substitute_probabilities = probabilities[utterance_idx, substitute_ids]
            possible_substitutes.append(
                [
                    {"word": word, "p": prob, "num_used": 0}
                    for word, prob in zip(substitutes, substitute_probabilities)
                ]
            )
        return possible_substitutes

    def apply_possible_substitutes(
        self,
        utterances: List[List[str]],
        slots: List[List[str]],
        possible_substitutes: List[Dict[int, List]],
    ) -> Tuple[Dict, Dict]:
        """
        Generate augmented utterances with given substitutes in possible positions

        Args:
            utterances: Tokenized utterances
            slots: Slot tags in BIO-format
            possible_substitutes: List of possible substitutes for each utterance.
                List element is a Dictionary that maps possible positions (for substitution) to generated substitutes.
        Returns:
            augmented_utterances: Dict that maps utterance number to generated utterances
            augmented_slots: Dict of corresponding tags
        """
        augmented_utterances, augmented_slots = defaultdict(list), defaultdict(list)
        for sample_idx, (utterance, tags, substitutes) in enumerate(
            zip(utterances, slots, possible_substitutes)
        ):
            target_ids = np.array(list(substitutes.keys()))
            num_augmented = 0
            while num_augmented < self.max_num_augmented_utters:
                # Pick positions to replace tokens
                mask = pick_random_target(
                    len(target_ids), proportion=1.0, max_num=self.num_replacements
                )
                replacement_ids = target_ids[mask].tolist()

                # For each picked position pick substitutes from possible substitutes
                replacement_words = []
                for idx in replacement_ids:
                    available_substitutes = [
                        (i, s["word"], s["p"])
                        for i, s in enumerate(substitutes[idx])
                        if s["used"] <= self.max_substitute_usage
                    ]
                    if not available_substitutes:
                        break
                    substitutes_ids, substitute_words, substitute_probs = list(
                        zip(*available_substitutes)
                    )
                    substitute_probs = np.array(substitute_probs)
                    substitute_probs /= substitute_probs.sum()
                    picked_subst_idx = np.random.choice(
                        substitutes_ids, size=1, p=substitute_probs
                    )[0]
                    substitutes[idx][picked_subst_idx]["used"] += 1
                    replacement_words.append(substitute_words[picked_subst_idx])

                if not replacement_words:
                    break

                new_tokens, new_slots = utterance.copy(), tags.copy()
                for idx, word in zip(replacement_ids, replacement_words):
                    if re.search("[-_/]", word):
                        word_tokens = re.split("[-_/]", word)
                        left_context, right_context = (
                            new_tokens[:idx],
                            new_tokens[idx + 1 :],
                        )
                        new_tokens = left_context + word_tokens + right_context

                        left_tags, right_tags = tags[:idx], tags[idx + 1 :]
                        target_slots = [tags[idx]]
                        if tags[idx].lower().startswith("b-"):
                            target_slots.extend(
                                [
                                    "i-" + tags[idx][2:]
                                    for _ in range(len(word_tokens) - 1)
                                ]
                            )
                        else:
                            target_slots.extend([tags[idx]] * (len(word_tokens) - 1))
                        new_slots = left_tags + target_slots + right_tags
                    else:
                        new_tokens[idx] = word
                augmented_utterances[sample_idx].append(" ".join(new_tokens))
                augmented_slots[sample_idx].append(" ".join(new_slots))
                num_augmented += 1
        return augmented_utterances, augmented_slots

    def augment_utterances(
        self, utterances: List[str], slots: Optional[List[str]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Main function which perform augmentation on given utterances.
        Additionally function can gets Slot BIO-tags.
        If slots are given, then function will replace only word with B- or I- tags.

        Args:
            utterances: List of input utterances,
            slots: Sequences of slots for each input utterance.
                Length (in tokens) of utterance and corresponding slot sequence must be equal.
        Returns:
            augmented_utterances: Dict that maps utterance number to generated utterances
            augmented_slots: Dict of corresponding tags
        """

        # Tokenize input utterance to pass it to generator
        utterances = [self.tokenizer(utterance) for utterance in utterances]

        # Get possible positions for generating substitutes
        if slots:
            slots = [self.slot_tokenizer(s) for s in slots]
            possible_positions = get_slot_positions(slots)
        else:
            slots = [["O"] * len(utterance) for utterance in utterances]
            possible_positions = [np.arange(len(utterance)) for utterance in utterances]

        # Pick target positions from possible positions
        possible_substitutes = []
        tokens_lists_for_generator, target_ids_for_generator = [], []
        for idx, positions in enumerate(possible_positions):
            mask = pick_random_target(
                sequence_length=len(positions),
                proportion=self.possibilities_proportion,
                max_num=self.max_num_possibilities,
            )
            picked_ids = positions[mask].tolist()
            possible_substitutes.append({idx: None for idx in picked_ids})
            tokens_lists_for_generator.extend([utterances[idx]] * len(picked_ids))
            target_ids_for_generator.extend(picked_ids)

        # Generate possible substitutions on picked target positions
        batch_reader = BatchReader(
            tokens_lists_for_generator,
            target_ids_for_generator,
            batch_size=self.batch_size,
        )
        utterance_idx = 0
        for tokens, targets in batch_reader:
            generated = self.generate_possible_substitutes(tokens, targets)
            for substitutes, target_idx in zip(generated, targets):
                if possible_substitutes[utterance_idx].get(target_idx, True):
                    utterance_idx += 1
                possible_substitutes[utterance_idx][target_idx] = substitutes
        return self.apply_possible_substitutes(utterances, slots, possible_substitutes)

    def augment_dataset(self, dataset_name: str, merge_with_origin: bool = True):
        """

        Args:
            dataset_name: name of augmenting dataset. It used for creating dataset reader.
            merge_with_origin: bool flag.
                If True then method merge augmented utterance with original in result DataFrame.
        Returns:
            augmented_dataset: pandas DataFrame with augmented utterances.
                DataFrame has 3 columns - 'utterance', 'intent', 'slots'
        """
        reader = NLUDatasetReader(dataset_name)
        dataset = reader.read_dataset()

        utterances = dataset.utterance.tolist()
        slots = dataset.slots.tolist()
        intents = dataset.intent.tolist()

        augmented_dataset = pd.DataFrame(columns=dataset.columns)
        batch_reader = BatchReader(
            utterances, slots, intents, batch_size=self.batch_size
        )
        for batch in tqdm(batch_reader):
            _utterances, _slots, _intents = batch
            # Augment batch of utterances
            augmented_utterances, augmented_slots = self.augment_utterances(
                _utterances, _slots
            )

            # Add augmented utterances to augmented dataset
            for sample_idx in augmented_utterances.keys():
                if merge_with_origin:
                    augmented_dataset = augmented_dataset.append(
                        {
                            "utterance": _utterances[sample_idx],
                            "slots": _slots[sample_idx],
                            "intent": _intents[sample_idx],
                        },
                        ignore_index=True,
                    )

                for utter, tags in zip(
                    augmented_utterances[sample_idx], augmented_slots[sample_idx]
                ):
                    augmented_dataset = augmented_dataset.append(
                        {
                            "utterance": utter,
                            "slots": tags,
                            "intent": _intents[sample_idx],
                        },
                        ignore_index=True,
                    )
        return augmented_dataset
