from typing import List, Dict, Optional, Tuple

from lexsubgen.utils.lemmatize import get_wordform2lemma


class SubstituteHandler:
    def __init__(
        self,
        lemmatizer: str = None,
        exclude_target: bool = False
    ):
        """
        Substitutes handler is a part of post-processing, but
        it processes predicted substitutes instead of processing
        probability distributions. It can lemmatize predicted substitutes and
        exclude the target word.

        Args:
            lemmatizer: defines which lemmatizer will process substitutes.
                All available types are in lexsubgen.utils.lemmatize.lemmatize_words
            exclude_target: if True target word will be removed from substitutes
        """
        self.lemmatizer = lemmatizer
        self.exclude_target = exclude_target
        self.prev_vocab = []

    def lemmatize(
        self,
        vocabulary: List[str],
        docs: List[List[str]]
    ) -> List[List[str]]:
        """
        Lemmatizes every word of every document
        Args:
            vocabulary: vocabulary of word forms
            docs: list of documents, each document is a list of words

        Returns: transformed documents

        """
        if not hasattr(self, "wordform2lemma") or self.prev_vocab != vocabulary:
            self.prev_vocab = vocabulary
            self.wordform2lemma = get_wordform2lemma(vocabulary, self.lemmatizer)

        return [
            [self.wordform2lemma[word_form] for word_form in doc]
            for doc in docs
        ]

    @staticmethod
    def exclude_words(
        docs: List[List[str]],
        to_exclude: List[Tuple[str]]
    ) -> List[List[str]]:
        """
        Excludes given words from documents
        Args:
            docs: list of documents, each document is a list of words
            to_exclude: each item in this list is a tuple of words
                that must be excluded from the corresponding documents

        Returns: transformed documents

        """
        transformed = []
        for doc, to_exclude_item in zip(docs, to_exclude):
            transformed.append([
                word for word in doc
                if word.lower() not in to_exclude_item
            ])
        return transformed

    def transform(
        self,
        substitutes: List[List[str]],
        word2id: Dict[str, int],
        target_words: List[str],
        target_pos: Optional[List[str]] = None,
        target_lemmas: Optional[List[str]] = None
    ) -> List[List[str]]:
        """
        Transforms substitutes and vocabulary leaving
        only lemmas of the words from vocabulary, predictions for words
        with the same lemma are aggregated according to the chosen strategy.

        Args:
            substitutes: predicted substitutes for target_words
            word2id: vocabulary
            target_words: list of target words
            target_pos: list of target part of speech tags (optional)
            target_lemmas: list of target lemmas (optional)

        Returns:
            transformed substitutes
        """
        vocabulary = list(word2id.keys())

        if self.lemmatizer is not None:
            substitutes = self.lemmatize(vocabulary, substitutes)

        if self.exclude_target:
            if target_lemmas is not None:
                to_exclude = [
                    (target_word.lower(), target_lemma.lower())
                    for target_word, target_lemma in zip(target_words, target_lemmas)
                ]
            else:
                to_exclude = [(target_word.lower(),) for target_word in target_words]

            substitutes = self.exclude_words(substitutes, to_exclude)

        return substitutes