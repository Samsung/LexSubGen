import pytest

from lexsubgen.metrics.candidate_ranking_metrics import gap_score


@pytest.fixture
def gold_substitutes():
    return ["intelligent", "clever", "i-am-mwe", "oov"]


@pytest.fixture
def gold_weights():
    return [3, 2, 1, 1]


# @pytest.fixture
# def model_prediction():
#     return [0.0581, 0.0872, 0.1918, 0.1162, 0.1453, 0.1744, 0.0988, 0.1279]


@pytest.fixture
def word2id():
    return {
        "happy": 0,
        "bright": 1,
        "positive": 2,
        "intelligent": 3,
        "clever": 4,
        "smart": 5,
        "talented": 6,
        "curious": 7,
    }


@pytest.fixture
def candidates():
    return ["positive", "intelligent", "smart", "clever", "talented", "i-am-mwe", "oov"]


@pytest.fixture
def ranked_candidates():
    return ["positive", "smart", "clever", "intelligent", "talented", "i-am-mwe", "oov"]


@pytest.fixture
def ranked_candidates_in_vocab():
    return ["positive", "smart", "clever", "intelligent", "talented"]


def test_gap(gold_substitutes, gold_weights, ranked_candidates_in_vocab, word2id):
    gap, _, _ = gap_score(
        gold_substitutes, gold_weights, ranked_candidates_in_vocab, word2id
    )
    assert gap == 0.2072072072072072


# def test_gap_ranked_candidates(
#     gold_substitutes,
#     gold_weights,
#     model_prediction,
#     word2id,
#     candidates,
#     ranked_candidates,
# ):
#     gap, _, _ = gap_score(
#         gold_substitutes,
#         gold_weights,
#         model_prediction,
#         word2id,
#         candidates,
#         return_ranked_candidates=True,
#     )
#     gap, ranked = gap
#     assert ranked == ranked_candidates


def test_gap_normalized(
    gold_substitutes, gold_weights, ranked_candidates_in_vocab, word2id
):
    _, gap_normalized, _ = gap_score(
        gold_substitutes, gold_weights, ranked_candidates_in_vocab, word2id
    )
    assert gap_normalized == 0.25555555555555554


def test_gap_vocab_normalized(
    gold_substitutes, gold_weights, ranked_candidates_in_vocab, word2id
):
    _, _, gap_vocab_normalized = gap_score(
        gold_substitutes, gold_weights, ranked_candidates_in_vocab, word2id
    )
    assert gap_vocab_normalized == 0.34848484848484845


# def test_all_vocab_gap(
#     gold_substitutes, gold_weights, model_prediction, word2id, candidates
# ):
#     gap, _, _ = all_vocab_gap_score(
#         gold_substitutes, gold_weights, model_prediction, word2id
#     )
#     assert gap == 0.18018018018018017
#
#
# def test_all_vocab_gap_normalized(
#     gold_substitutes, gold_weights, model_prediction, word2id, candidates
# ):
#     _, gap_normalized, _ = all_vocab_gap_score(
#         gold_substitutes, gold_weights, model_prediction, word2id
#     )
#     assert gap_normalized == 0.2222222222222222
#
#
# def test_all_vocab_gap_vocab_normalized(
#     gold_substitutes, gold_weights, model_prediction, word2id, candidates
# ):
#     _, _, gap_vocab_normalized = all_vocab_gap_score(
#         gold_substitutes, gold_weights, model_prediction, word2id
#     )
#     assert gap_vocab_normalized == 0.303030303030303
