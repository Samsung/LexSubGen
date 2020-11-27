import pytest
from pathlib import Path
from lexsubgen.evaluations.lexsub import LexSubEvaluation
from lexsubgen import SubstituteGenerator
from lexsubgen.datasets.base_reader import DatasetReader

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


@pytest.fixture
def bert_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "bert.jsonnet")
    )


@pytest.fixture
def bert_embs_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "bert_embs.jsonnet")
    )


@pytest.fixture
def coinco_dataset_reader():
    return DatasetReader.from_config(
        str(CONFIGS_PATH / "dataset_readers" / "lexsub" / "coinco.jsonnet")
    )


@pytest.fixture
def semeval_all_dataset_reader():
    return DatasetReader.from_config(
        str(CONFIGS_PATH / "dataset_readers" / "lexsub" / "semeval_all.jsonnet")
    )


def test_semeval_all_bert(bert_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/bert.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_bert'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_bert'
    """
    scores = LexSubEvaluation(
        substitute_generator=bert_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(54.42, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(38.39), str(scores)
    assert scores["prec@3"] == pytest.approx(27.73), str(scores)
    assert scores["rec@10"] == pytest.approx(39.57), str(scores)
    assert scores["recall"] == pytest.approx(74.64), str(scores)


def test_coinco_bert(bert_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/bert.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_bert'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_bert'
    """
    scores = LexSubEvaluation(
        substitute_generator=bert_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(50.5, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(42.56), str(scores)
    assert scores["prec@3"] == pytest.approx(32.64), str(scores)
    assert scores["rec@10"] == pytest.approx(28.73), str(scores)


def test_semeval_all_bert_embs(bert_embs_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/bert_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_bert_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_bert_embs'
    """
    scores = LexSubEvaluation(
        substitute_generator=bert_embs_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(53.87, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(41.64), str(scores)
    assert scores["prec@3"] == pytest.approx(30.59), str(scores)
    assert scores["rec@10"] == pytest.approx(43.88), str(scores)


def test_coinco_bert_embs(bert_embs_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/bert_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_bert_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_bert_embs'
    """
    scores = LexSubEvaluation(
        substitute_generator=bert_embs_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(50.85, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(46.05), str(scores)
    assert scores["prec@3"] == pytest.approx(35.63), str(scores)
    assert scores["rec@10"] == pytest.approx(31.67), str(scores)
