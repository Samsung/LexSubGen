import logging
import os
from itertools import chain
from pathlib import Path
from typing import List, Set, Dict, Tuple
from xml.etree import ElementTree
from word_forms.word_forms import get_word_forms

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

from lexsubgen.datasets.lexsub import DatasetReader
from lexsubgen.datasets.utils import download_dataset
from lexsubgen.utils.register import CACHE_DIR
from lexsubgen.utils.wsi import SEMEVAL2013URL, SEMEVAL2010URL, SEMEVAL2010TESTURL


logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)


def wsi_logging_info(_logger):
    """
    Decorates read_dataset methods for WSI readers.
    Adds logging information about the read dataset
    Args:
        _logger: an object that allows to call .info method

    Returns: decorator for read_dataset method
    """
    def decorator(wsi_reader_method):
        def wrapper(self, *args, **kwargs):
            _logger.info(f"Reading {self.dataset_name} dataset")
            df, gold_labels, gold_labels_path = wsi_reader_method(self, *args, **kwargs)
            _logger.info(f"Reading done. {self.dataset_name} dataset "
                        f"contains {len(df)} lines, "
                        f"{len(df.group_by.unique())} ambigious words "
                        f"and following {df.pos_tag.unique()} POS tags")
            return df, gold_labels, gold_labels_path
        return wrapper
    return decorator


class WSIDatasetReader(DatasetReader):
    data_root_path = Path(CACHE_DIR) / "wsi"
    df_columns = ['context_id', 'group_by', 'target_lemma', 'pos_tag', 'sentence', 'target_id']

    def __init__(self, dataset_name: str, data_root_path: str, url: str):
        super(WSIDatasetReader, self).__init__(
            dataset_name=dataset_name,
            data_root_path=data_root_path,
            url=url
        )

    @staticmethod
    def read_gold_labels(gold_labels_path: str) -> Dict:
        gold_labels = dict()
        with open(gold_labels_path, 'r') as f:
            for instance in f:
                _, context_id, *clusters = instance.strip().split(" ")
                # TODO: gold labels might consist of more than 1 cluster label
                gold_labels[context_id] = clusters[0]
        return gold_labels


class SemEval2013DatasetReader(WSIDatasetReader):
    dataset_name = "semeval-2013"
    inner_path = (
        Path("SemEval-2013-Task-13-test-data")
        / "contexts"
        / "senseval2-format"
        / "semeval-2013-task-13-test-data.senseval2.xml"
    )
    gold_labels_path = (
        Path("SemEval-2013-Task-13-test-data") / "keys" / "gold" / "all.key"
    )

    def __init__(self):
        super(SemEval2013DatasetReader, self).__init__(
            dataset_name=self.dataset_name,
            data_root_path=self.data_root_path,
            url=SEMEVAL2013URL
        )

    @wsi_logging_info(logger)
    def read_dataset(self) -> Tuple[pd.DataFrame, Dict[str, str], Path]:
        """
        Reads SemEval2013 task 13 dataset. This dataset is stored in xml format where
        for each target word the context is given and also its part of speech tag.

        Returns:
            df: pandas DataFrame that contains following columns:
                'instance_id', 'target_word', 'pos_tag', 'sentence', 'target_id'
        """
        data_path = self.data_root_path / self.dataset_name
        gold_labels = self.read_gold_labels(data_path / self.gold_labels_path)

        xml_dataset_path = data_path / self.inner_path
        corpus = ElementTree.parse(xml_dataset_path).getroot()
        dataset_list = []
        for lexelt in corpus:
            group_by = lexelt.attrib["item"]
            target_lemma, pos_tag = group_by.split(".")
            for instance in lexelt:
                context_id = instance.attrib["id"]
                context = [ctx for ctx in instance][0]
                lctx, target, rctx = [text.strip() for text in context.itertext()]
                lctx_tokens = nltk.word_tokenize(lctx)
                rctx_tokens = nltk.word_tokenize(rctx)
                sentence = lctx_tokens + [target] + rctx_tokens
                target_idx = len(lctx_tokens)
                # TODO: gold labels file does not
                #  contain several instances from the dataset
                if context_id in gold_labels:
                    dataset_list.append((
                        context_id, group_by, target_lemma, pos_tag, sentence, target_idx
                    ))

        dataset_df = pd.DataFrame(dataset_list, columns=self.df_columns)
        assert len(dataset_df.context_id.unique()) == len(dataset_df)

        return dataset_df, gold_labels, data_path / self.gold_labels_path


class SemEval2010DatasetReader(WSIDatasetReader):
    dataset_name = "semeval-2010"
    inner_path = Path(dataset_name) / "test_data"
    gold_labels_path = (
        Path(dataset_name) / "evaluation" / "unsup_eval" / "keys" / "all.key"
    )
    lemma2form = {
        "figure": ["figger", "figgered"],
        "straighten": ["half-straightened"],
        "lie": ["lah"],
    }

    def __init__(self, use_surrounding_context: bool = True):
        super(SemEval2010DatasetReader, self).__init__(
            dataset_name=self.dataset_name,
            data_root_path=self.data_root_path,
            url=SEMEVAL2010URL
        )

        self.use_surrounding_context = use_surrounding_context

        self.test_data_path = self.data_root_path / self.inner_path
        if not os.path.exists(self.test_data_path):
            download_dataset(
                SEMEVAL2010TESTURL,
                self.data_root_path / self.dataset_name
            )

    @staticmethod
    def _find_target_word_idx(tokens: List[str], word_forms: Set):
        for idx, token in enumerate(tokens):
            token_lower = token.lower()
            lemmas = {lemma for pos in ['v', 'n', 'a', 'r']
                      for lemma in wn._morphy(token_lower, pos)}
            if lemmas.intersection(word_forms) or token_lower in word_forms:
                return idx
        raise ValueError(f"Target word was not found {tokens}\n{word_forms}")

    @wsi_logging_info(logger)
    def read_dataset(self):
        gold_labels = self.read_gold_labels(
            self.data_root_path / self.gold_labels_path
        )

        paths = [
            self.test_data_path / "nouns" / file
            for file in os.listdir(self.test_data_path / "nouns")
        ]

        paths.extend([
            self.test_data_path / "verbs" / file
            for file in os.listdir(self.test_data_path / "verbs")
        ])

        dataset_list = []
        for path in paths:
            corpus = ElementTree.parse(path).getroot()
            target_lemma, pos_tag, _ = corpus.tag.split('.')
            group_by = f"{target_lemma}.{pos_tag}"
            lemma2unknown_form = self.lemma2form.get(target_lemma.lower(), [])
            target_word_forms = {
                elem
                for _set in get_word_forms(target_lemma.lower()).values()
                for elem in chain(_set, lemma2unknown_form)
            }

            for instance in corpus:
                context_id = instance.tag
                target_sentences = [s.text.strip()
                                    for s in instance.iter("TargetSentence")]
                assert len(target_sentences) == 1, ("Something went wrong. "
                                                    "Number of Target Sentences should be 1")
                target_sentence = target_sentences[0]
                surrounding_context = [s.strip() for s in instance.itertext()]

                if len(surrounding_context) == 3 and surrounding_context[1] == target_sentence:
                    # ['left', 'target', 'right']
                    left_context, right_context = surrounding_context[0], surrounding_context[2]
                elif len(surrounding_context) == 2 and surrounding_context[0] == target_sentence:
                    # ['target', 'right']
                    left_context, right_context = "", surrounding_context[1]
                elif len(surrounding_context) == 2 and surrounding_context[1] == target_sentence:
                    # ['left', 'target']
                    left_context, right_context = surrounding_context[0], ""
                elif len(surrounding_context) == 1 and surrounding_context[0] == target_sentence:
                    # ['target']
                    left_context, right_context = "", ""
                else:
                    raise ValueError(
                        "Something went wrong. Number of Target Sentences should be 1"
                    )

                tokens = target_sentence.split()
                target_idx = self._find_target_word_idx(
                    tokens, target_word_forms
                )

                if self.use_surrounding_context:
                    l, r = left_context.split(), right_context.split()
                    target_idx += len(l)
                    tokens = l + tokens + r

                dataset_list.append((
                    context_id, group_by, target_lemma, pos_tag, tokens, target_idx
                ))

        dataset_df = pd.DataFrame(dataset_list, columns=self.df_columns)
        assert len(dataset_df.context_id.unique()) == len(dataset_df)

        return dataset_df, gold_labels, self.data_root_path / self.gold_labels_path
