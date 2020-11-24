import pytest

import numpy as np

from lexsubgen.post_processors.base_postprocessor import LowercasePostProcessor


@pytest.fixture()
def log_probs():
    log_probs = np.array(
        [
            [0.4, 0.2, 0.3, 0.1, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.6],
            [0.6, 0.7, 0.8, 0.9, 0.2],
        ]
    )
    return log_probs


@pytest.fixture()
def word2id():
    return {"test": 0, "Kek": 1, "kek": 2, "Test": 3, "Cheburek": 4}


@pytest.fixture()
def target_words():
    return ["kek", "Cheburek"]


def log_sum_probs(logit1, logit2):
    return np.log(np.exp(logit1) + np.exp(logit2))


def test_lowercase_postprocessor_sum(log_probs, word2id, target_words):
    post_processor = LowercasePostProcessor(strategy="sum")
    tr_log_probs, tr_word2id = post_processor.transform(
        log_probs=log_probs, word2id=word2id, target_words=target_words
    )
    assert not set(tr_word2id.keys()).difference({"test", "kek", "cheburek"})
    true_log_probs = np.array(
        [
            [log_sum_probs(0.4, 0.1), log_sum_probs(0.2, 0.3), 0.5],
            [log_sum_probs(0.2, 0.5), log_sum_probs(0.4, 0.3), 0.6],
            [log_sum_probs(0.6, 0.9), log_sum_probs(0.7, 0.8), 0.2],
        ]
    )
    assert np.allclose(true_log_probs, tr_log_probs)


def test_lowercase_postprocessor_max(log_probs, word2id, target_words):
    post_processor = LowercasePostProcessor(strategy="max")
    tr_log_probs, tr_word2id = post_processor.transform(
        log_probs=log_probs, word2id=word2id, target_words=target_words
    )
    assert not set(tr_word2id.keys()).difference({"test", "kek", "cheburek"})
    true_log_probs = np.array([[0.4, 0.3, 0.5], [0.5, 0.4, 0.6], [0.9, 0.8, 0.2]])
    assert np.allclose(true_log_probs, tr_log_probs)


def test_lowercase_postprocessor_assert():
    with pytest.raises(ValueError):
        post_processor = LowercasePostProcessor(strategy="exp")
