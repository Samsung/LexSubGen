import re
import warnings
from collections import defaultdict
from multiprocessing import cpu_count
from typing import List, Dict, Tuple, Union

import numpy as np
import spacy
from nltk.stem import WordNetLemmatizer
from spacy.lang.en import English
from tqdm import tqdm

from lexsubgen.utils.register import memory
from lexsubgen.utils.wordnet_relation import to_wordnet_pos

# import pymorphy2
# from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES, LOOKUP


to_spacy_pos = {
    "n": "NOUN",
    "a": "ADJ",
    "v": "VERB",
    "r": "ADV",
    "a.n": "NOUN",
    "n.v": "VERB",
    "n.a": "ADJ",
    "J": "ADJ",
    "V": "VERB",
    "R": "ADV",
    "N": "NOUN",
}


@memory.cache
def spacy_lemmatize(
    unlem: List[str], pos_tag: Union[str, List[str]] = "NOUN", verbose: bool = False
) -> List[str]:
    """
    Lemmatize sequence of words with Spacy lemmatizer.

    Args:
        unlem: sequence of unlemmatized words
        pos_tag: part-of-speech tags of words
            if str than this part-of-speech tag will be used with all words
        verbose: whether to print misc information

    Returns:
        sequence of lemmatized words
    """
    pattern = re.compile(r"[#\[-]")
    lemmatizer = English.Defaults.create_lemmatizer()
    # lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES, LOOKUP)
    gen = unlem
    if verbose:
        gen = tqdm(unlem, desc="Vocabulary Lemmatization")

    if isinstance(pos_tag, str):
        pos_tag = [to_spacy_pos.get(pos_tag, "NOUN")] * len(unlem)
    else:
        pos_tag = [to_spacy_pos.get(pos_tag_, "NOUN") for pos_tag_ in pos_tag]

    new_vocab = [
        word if pattern.match(word) else lemmatizer(word, pos_tag_)[0]
        for word, pos_tag_ in zip(gen, pos_tag)
    ]
    return new_vocab


@memory.cache
def old_spacy_lemmatize(unlem: List[str], verbose: bool = False) -> List[str]:
    """
    Lemmatize sequence of words with Spacy pipeline.

    Args:
        unlem: sequence of unlemmatized words
        verbose: whether to print misc information

    Returns:
        sequence of lemmatized words
    """

    nlp = spacy.load("en", disable=["ner", "parser"])

    lemmatized_words = []

    with warnings.catch_warnings(record=True) as wn:
        # When using spacy 2.1.8 it warns: "DeprecationWarning: [W016] The keyword argument `n_threads` is now deprecated.
        #   As of v2.2.2, the argument `n_process` controls parallel inference via multiprocessing."
        warnings.simplefilter("ignore")
        gen = zip(nlp.pipe(unlem, batch_size=1000, n_threads=cpu_count()), unlem)

        if verbose:
            gen = tqdm(gen, total=len(unlem), desc=f"Lemmatization of {len(unlem)} words")

        for spacyed, word in gen:
            if "#" in word or "[" in word or word == "":
                lemma = word
            else:
                lemma = (
                    spacyed[0].lemma_
                    if spacyed[0].lemma_ != "-PRON-"
                    else spacyed[0].lower_
                )
            lemmatized_words.append(lemma)

        # Checking if it is not just DeprecationWarning
        assert len(wn) <= 1, str(wn)
        assert len(wn) <= 1 or issubclass(wn[-1].category, DeprecationWarning), str(wn)
        assert len(wn) <= 1 or "deprecated" in str(wn[-1].message), str(wn)

    return lemmatized_words


@memory.cache
def nltk_lemmatize(
    unlem: List[str], pos_tag: Union[str, List[str]] = "n", verbose: bool = False
) -> List[str]:
    """
    Lemmatize sequence of words with nltk tokenizer.

    Args:
        unlem: sequence of unlemmatized words
        pos_tag: part-of-speech tags of words
            if str than this part-of-speech tag will be used with all words
        verbose: whether to print misc information

    Returns:
        sequence of lemmatized words

    """
    pattern = re.compile(r"[#\[-]")
    lemmatizer = WordNetLemmatizer()
    gen = unlem
    if verbose:
        gen = tqdm(unlem, desc="Vocabulary Lemmatization")

    # convert to appropriate pos abbreviation
    if isinstance(pos_tag, str):
        pos_tag = [to_wordnet_pos.get(pos_tag, "n")] * len(unlem)
    else:
        pos_tag = [to_wordnet_pos.get(pos_tag_, "n") for pos_tag_ in pos_tag]

    new_vocab = [
        word if pattern.match(word) else lemmatizer.lemmatize(word, pos_tag_)
        for word, pos_tag_ in zip(gen, pos_tag)
    ]
    return new_vocab


@memory.cache
def pymorphy_ru_lemmatize(unlem: List[str], verbose: bool = False) -> List[str]:
    """
    Lemmatizes sequence of words with Pymorphy lemmatizer.

    Args:
        unlem: sequence of unlemmatized words
        verbose: whether to print misc information

    Returns:
        sequence of lemmatized words
    """
    lemmatizer = pymorphy2.MorphAnalyzer()
    gen = unlem
    if verbose:
        gen = tqdm(unlem, desc='Vocabulary Lemmatization')

    new_vocab = [word if ('#' in word or '[' in word)
                 else lemmatizer.parse(word)[0].normal_form
                 for word in gen]

    return new_vocab


def lemmatize_words(
    unlem: List[str],
    lemmatizer_name: str,
    pos_tag: Union[str, List[str]] = "n",
    verbose: bool = False,
) -> List[str]:
    """
    This function just chooses right lemmatizer that is specified by name.

    Args:
        unlem: sequence of unlemmatized words
        lemmatizer_name: name of the lemmatizer (currently supported lemmatizers are nltk and Spacy).
        pos_tag: part-of-speech tags of words
            if str than this part-of-speech tag will be used with all words
        verbose: whether to print misc information

    Returns:
        sequence of lemmatized words
    """
    if lemmatizer_name == "nltk":
        lemmatized = nltk_lemmatize(unlem, pos_tag, verbose)
    elif lemmatizer_name == "spacy":
        lemmatized = spacy_lemmatize(unlem, pos_tag, verbose)
    elif lemmatizer_name == "spacy_old":
        lemmatized = old_spacy_lemmatize(unlem, verbose)
    elif lemmatizer_name == "pymorphy-ru":
        lemmatized = pymorphy_ru_lemmatize(unlem, verbose)
    else:
        raise ValueError(f"Incorrect lemmatizer type: {lemmatizer_name}")
    return lemmatized


def lemmatize_batch(
    probs: np.ndarray,
    forms_ids_lists: List[List[int]],
    strategy: str = "max",
    parallel: bool = False,
) -> np.ndarray:
    """
    Aggregates probabilities of different word forms to their lemmas.
    Different aggregation strategies could be chosen, currently we support
    taking maximum probability of all word forms and summing them.

    Args:
        probs: matrix of distributions over vocabulary for each batch instance
        forms_ids_lists: list of indexes of word forms for a lemma
        strategy: aggregation strategy (max or sum)
        parallel: whether to aggregate data for different words in parallel (default: False)

    Returns:
        new probability distributions with aggregate probabilities for lemmas
    """
    assert strategy == "max" or strategy == "sum"
    new_batch = np.zeros((probs.shape[0], 0))
    if not parallel:
        new_batches = []
        if forms_ids_lists:
            for i, forms_ids in enumerate(forms_ids_lists):
                new_batches.append(
                    np.__getattribute__(strategy)(
                        probs[:, forms_ids], axis=1, keepdims=True
                    )
                )
            new_batch = np.concatenate(new_batches, axis=1)
    else:
        new_batch = probs[:, forms_ids_lists].__getattribute__(strategy)(axis=-1)
    return new_batch


@memory.cache
def get_all_vocabs(
    old_word2id: Dict[str, int],
    lemmatizer: str,
    pos_tag: str = "n",
    verbose: bool = False,
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    Method that lemmatizes a vocabulary with the chosen lemmatizer.
    So the original vocabulary shrinks to vocabulary of their lemmas.

    Args:
        old_word2id: old vocabulary with unlemmatized words
        lemmatizer: name of the lemmatizer to be used for processing
        verbose: whether to print misc information

    Returns:
        mapping from lemmas to their word forms,
        mapping from words to indexes (new vocabulary with lemmatized words)
    """
    sorted_vocab = sorted(old_word2id.items(), key=lambda x: x[0])
    sorted_words, sorted_idxs = list(zip(*sorted_vocab))

    new_vocab = lemmatize_words(sorted_words, lemmatizer, pos_tag, verbose)

    lemma2words = defaultdict(list)
    word2id = dict()
    for word, old_idx, lemma in zip(sorted_words, sorted_idxs, new_vocab):
        lemma2words[lemma].append(old_idx)
        word2id[lemma] = word2id.get(lemma, len(word2id))

    return lemma2words, word2id


@memory.cache
def get_wordform2lemma(
    vocabulary: Dict[str, int],
    lemmatizer: str,
    pos_tag: str = "n",
    verbose: bool = False
):
    """
    Wordform2lemma is a dict that maps word forms to its lemmas
    Args:
        word2id: vocabulary of word forms

    Returns: mapping after lemmatization of the given vocabulary
    """
    lemmatized = lemmatize_words(vocabulary, lemmatizer, pos_tag, verbose)
    return dict(zip(vocabulary, lemmatized))
