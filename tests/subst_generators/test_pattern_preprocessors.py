import pytest

from lexsubgen.pre_processors.pattern_preprocessors import (
    PatternPreprocessor,
    insert_pattern,
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


@pytest.fixture()
def pattern():
    return "{target} ( or even {predict} )"


def validate_sizes(sentences1, sentences2, target_ids1, target_ids2):
    assert len(sentences2) == len(sentences1)
    assert len(sentences2) == len(target_ids2)
    assert len(target_ids2) == len(target_ids1)


def test_insert_pattern(sentences, target_ids, pattern):
    for sentence, target_id in zip(sentences, target_ids):
        mod_sentence, mod_target_id = insert_pattern(sentence, target_id, pattern)
        target_word = sentence[target_id]
        pattern_start = mod_sentence[target_id:]

        assert pattern_start[0] == target_word
        assert pattern_start[1] == "("
        assert pattern_start[2] == "or"
        assert pattern_start[3] == "even"
        assert pattern_start[4] == target_word
        assert pattern_start[5] == ")"


def test_pattern_preprocessor(sentences, target_ids, pattern):
    preprocessor = PatternPreprocessor(pattern, lowercase=False)

    tr_sentences, tr_target_ids = preprocessor.transform(sentences, target_ids)
    validate_sizes(sentences, tr_sentences, target_ids, tr_target_ids)


def test_lowercase(sentences, target_ids, pattern):
    preprocessor = PatternPreprocessor(pattern, lowercase=True)

    tr_sentences, tr_target_ids = preprocessor.transform(sentences, target_ids)
    validate_sizes(sentences, tr_sentences, target_ids, tr_target_ids)

    for sentence in tr_sentences:
        assert all([(not tok.isalpha()) or tok.islower() for tok in sentence])
