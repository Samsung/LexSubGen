import string
from typing import List, Tuple, Optional

from overrides import overrides

PADDING_TEXT = (
    "In 1991, the remains of Russian Tsar Nicholas II and "
    "his family (except for Alexei and Maria) are discovered. "
    "The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, "
    "narrates the remainder of the story. 1883 Western Siberia, a young "
    "Grigori Rasputin is asked by his father and a group of men to "
    "perform magic. Rasputin has a vision and denounces one of the men "
    "as a horse thief. Although his father initially slaps him for making "
    "such an accusation, Rasputin watches as the man is chased outside "
    "and beaten. Twenty years later, Rasputin sees a vision of the Virgin Mary, "
    "prompting him to become a priest. Rasputin quickly becomes famous, with people, "
    "even a bishop, begging for his blessing. "
)


class Preprocessor:
    def __init__(self):
        """
        Base class for pre-processing modules.
        """
        pass

    def transform(
        self, sentences: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Method that transforms instances of lexical substitution task.
        Must be implemented in the child classes.

        Args:
            sentences: list of sentences which are represented as a token sequence
            target_ids: indexes of target word in sentence

        Returns:
            transformed sentences list and new target indexes
        """
        raise NotImplemented


class LowerCasePreprocessor(Preprocessor):
    """
    Preprocessor that transforms sentence to lower case.
    """

    @overrides
    def transform(
        self, sentences: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Transforms sentence tokens into lower case register.

        Args:
            sentences: list of sentences which are represented as a token sequence
            target_ids: indexes of target word in sentence

        Returns:
            lower-cased sentences list and non-changed target indexes
        """
        return (
            [[token.lower() for token in sentence] for sentence in sentences],
            target_ids,
        )


class AddPunctPreprocessor(Preprocessor):
    """
    Preprocessor that adds punctuation at the end of a sentence is there isn't one.
    """

    @overrides
    def transform(
        self, sentences: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Adds dot at the end of a sentence if there is no punctuation symbol at the end.

        Args:
            sentences: list of sentences which are represented as a token sequence
            target_ids: indexes of target word in sentence

        Returns:
            transformed sentences list and non-changed target indexes
        """
        transformed_sentences = []
        for sentence in sentences:
            if sentence[-1] not in string.punctuation:
                sentence.append(".")
            transformed_sentences.append(sentence)
        return transformed_sentences, target_ids


class TitlePreprocessor(Preprocessor):
    """
    Preprocessor that title the first word in a sentence.
    """

    @overrides
    def transform(
        self, sentences: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Title first word in a sentences.

        Args:
            sentences: list of sentences which are represented as a token sequence
            target_ids: indexes of target word in sentence

        Returns:
            transformed sentences list and non-changed target indexes
        """
        transformed_sentences = []
        for sentence in sentences:
            sentence[0] = sentence[0].title()
            transformed_sentences.append(sentence)
        return transformed_sentences, target_ids


class AndPreprocessor(Preprocessor):
    def __init__(self, cased: bool = True):
        """
        Preprocessor that adds and token if target word is the first in a sentence.

        Args:
            cased: whether to use case version of and or not.
        """
        super().__init__()
        self.and_token = "And" if cased else "and"

    @overrides
    def transform(
        self, sentences: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Adds 'And' or 'and' at the sentence start if target word is the first.

        Args:
            sentences: list of sentences which are represented as a token sequence
            target_ids: indexes of target word in sentence

        Returns:
            transformed sentences list and new target indexes
        """
        transformed_sentences = []
        transformed_target_ids = []
        for sentence, target_id in zip(sentences, target_ids):
            if target_id == 0:
                sentence = [self.and_token, sentence[0].lower()] + sentence[1:]
                target_id += 1
            transformed_sentences.append(sentence)
            transformed_target_ids.append(target_id)
        return transformed_sentences, transformed_target_ids


class PadTextPreprocessor(Preprocessor):
    # TODO: add some tokenization not simple split. Different tokenization for different models.
    def __init__(self, text: Optional[str] = None, special_end_token: Optional[str] = None):
        """
        Preprocessor that pads original sentence with some pre-defined text.

        Args:
            text: text that would be padded to the original sentence.
            special_end_token: this token would be added right at the end of a tokenized text, e.g. <eod> for XLNet.
        """
        super().__init__()
        text = text or PADDING_TEXT
        self.text_tokens = text.split()
        if special_end_token is not None:
            self.text_tokens.append(special_end_token)
        self.text_len = len(self.text_tokens)

    @overrides
    def transform(
        self, sentences: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Adds padding text at the start of a sentence.

        Args:
            sentences: list of sentences which are represented as a token sequence
            target_ids: indexes of target word in sentence

        Returns:
            transformed sentences list and new target indexes
        """
        transformed_sentences = []
        transformed_target_ids = []
        for sentence, target_id in zip(sentences, target_ids):
            transformed_sentences.append(self.text_tokens + sentence)
            transformed_target_ids.append(target_id + self.text_len)
        return transformed_sentences, transformed_target_ids


class CopyPreprocessor(Preprocessor):
    def __init__(self, sep_text: Optional[str] = None):
        """
        Preprocessor that prepends original sentence with its copy.
        Its a way to inject information about a target to a model.

        Args:
            sep_text: optional text that would be added between a sentence and its copy
        """
        super().__init__()
        self.special_end_text = sep_text

    @overrides
    def transform(
        self, sentences: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Prepends sentences with their copies.

        Args:
            sentences: list of sentences which are represented as a token sequence
            target_ids: indexes of target word in sentence

        Returns:
            transformed sentences list and new target indexes
        """
        transformed_sentences = []
        transformed_target_ids = []
        for sentence, target_id in zip(sentences, target_ids):
            transformed_sentences.append(sentence + sentence)
            transformed_target_ids.append(target_id + len(sentence))
        return transformed_sentences, transformed_target_ids


class CopyTwicePreprocessor(Preprocessor):
    def __init__(self, sep_text: Optional[str] = None):
        """
        Preprocessor that prepends original sentence with its copy.
        Its a way to inject information about a target to a model.

        Args:
            sep_text: optional text that would be added between a sentence and its copy
        """
        super().__init__()
        self.special_end_text = sep_text

    @overrides
    def transform(
        self, sentences: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Prepends sentences with their copies.

        Args:
            sentences: list of sentences which are represented as a token sequence
            target_ids: indexes of target word in sentence

        Returns:
            transformed sentences list and new target indexes
        """
        transformed_sentences = []
        transformed_target_ids = []
        for sentence, target_id in zip(sentences, target_ids):
            transformed_sentences.append(sentence + sentence + sentence)
            transformed_target_ids.append(target_id + len(sentence))
        return transformed_sentences, transformed_target_ids
