import pytest
import numpy as np

from lexsubgen.prob_estimators.xlnet_estimator import XLNetProbEstimator


@pytest.fixture()
def tokens_lists():
    return [["I", "love", "cats"]]


@pytest.fixture()
def target_idxs():
    return [2]


def test_get_log_probs(tokens_lists, target_idxs):
    xlnet_prob_est = XLNetProbEstimator()
    logits, word2id = xlnet_prob_est.get_log_probs(tokens_lists, target_idxs)
    assert isinstance(logits, np.ndarray)
    assert isinstance(word2id, dict)
    assert logits.shape[1] == len(word2id)
    assert logits.shape[0] == 1
