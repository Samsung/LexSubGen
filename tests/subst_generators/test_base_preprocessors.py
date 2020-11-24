import string

import pytest

from lexsubgen.pre_processors.base_preprocessors import (
    LowerCasePreprocessor,
    TitlePreprocessor,
    AndPreprocessor,
    AddPunctPreprocessor,
    PadTextPreprocessor,
    CopyPreprocessor,
)


@pytest.fixture()
def sentences():
    return [
        "finally , cat sitting on a mat".split(),
        "He was a bright boy who lived near our house .".split(),
        "Hello , how are you , John ?".split(),
        "We 've played volleyball for a long time".split(),
        "Additionally , he had a big suitcase with candies .".split(),
    ]


@pytest.fixture()
def target_ids():
    return [0, 3, 6, 3, 0]


def validate_sizes(sentences1, sentences2, target_ids1, target_ids2):
    assert len(sentences2) == len(sentences1)
    assert len(sentences2) == len(target_ids2)
    assert len(target_ids2) == len(target_ids1)


def test_lowercase(sentences, target_ids):
    preprocessor = LowerCasePreprocessor()
    tr_sentences, tr_target_ids = preprocessor.transform(sentences, target_ids)

    validate_sizes(sentences, tr_sentences, target_ids, tr_target_ids)
    assert all([id1 == id2 for id1, id2 in zip(target_ids, tr_target_ids)])

    for sentence in tr_sentences:
        assert all([(not tok.isalpha()) or tok.islower() for tok in sentence])


def test_title(sentences, target_ids):
    preprocessor = TitlePreprocessor()
    tr_sentences, tr_target_ids = preprocessor.transform(sentences, target_ids)

    validate_sizes(sentences, tr_sentences, target_ids, tr_target_ids)
    assert all([id1 == id2 for id1, id2 in zip(target_ids, tr_target_ids)])

    for sentence in tr_sentences:
        assert sentence[0].istitle(), f"Not titled {sentence[0]}"


@pytest.mark.parametrize("cased", (True, False))
def test_add_and(sentences, target_ids, cased):
    preprocessor = AndPreprocessor(cased=cased)
    tr_sentences, tr_target_ids = preprocessor.transform(sentences, target_ids)

    validate_sizes(sentences, tr_sentences, target_ids, tr_target_ids)

    original_targets = [
        sentence[target_id] for sentence, target_id in zip(sentences, target_ids)
    ]
    tr_targets = [
        sentence[target_id] for sentence, target_id in zip(tr_sentences, tr_target_ids)
    ]
    assert len(original_targets) == len(tr_targets)
    assert all(
        [
            tok1.lower() == tok2.lower()
            for tok1, tok2 in zip(original_targets, tr_targets)
        ]
    )

    for sentence, target_id in zip(tr_sentences, target_ids):
        if target_id == 0:
            if cased:
                assert sentence[0] == "And"
            else:
                assert sentence[0] == "and"
        assert (not sentence[1].isalpha()) or sentence[1].islower()


def test_add_punctuation(sentences, target_ids):
    preprocessor = AddPunctPreprocessor()
    tr_sentences, tr_target_ids = preprocessor.transform(sentences, target_ids)

    validate_sizes(sentences, tr_sentences, target_ids, tr_target_ids)
    assert all([id1 == id2 for id1, id2 in zip(target_ids, tr_target_ids)])

    for sentence in tr_sentences:
        assert sentence[-1] in string.punctuation


def test_pad_text(sentences, target_ids):
    preprocessor = PadTextPreprocessor()
    tr_sentences, tr_target_ids = preprocessor.transform(sentences, target_ids)

    validate_sizes(sentences, tr_sentences, target_ids, tr_target_ids)
    assert all(
        [
            len(sentence1) < len(sentence2)
            for sentence1, sentence2 in zip(sentences, tr_sentences)
        ]
    )
    assert all(
        [
            target_id1 < target_id2
            for target_id1, target_id2 in zip(target_ids, tr_target_ids)
        ]
    )

    original_targets = [
        sentence[target_id] for sentence, target_id in zip(sentences, target_ids)
    ]
    tr_targets = [
        sentence[target_id] for sentence, target_id in zip(tr_sentences, tr_target_ids)
    ]
    assert all([tok1 == tok2 for tok1, tok2 in zip(original_targets, tr_targets)])


def test_copy(sentences, target_ids):
    preprocessor = CopyPreprocessor()
    tr_sentences, tr_target_ids = preprocessor.transform(sentences, target_ids)

    validate_sizes(sentences, tr_sentences, target_ids, tr_target_ids)

    original_targets = [
        sentence[target_id] for sentence, target_id in zip(sentences, target_ids)
    ]
    tr_targets = [
        sentence[target_id] for sentence, target_id in zip(tr_sentences, tr_target_ids)
    ]
    assert all([tok1 == tok2 for tok1, tok2 in zip(original_targets, tr_targets)])

    for tr_sentence, sentence in zip(tr_sentences, sentences):
        assert len(tr_sentence) == 2 * len(sentence)
        assert tr_sentence[: len(sentence)] == sentence
