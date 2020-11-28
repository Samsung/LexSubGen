import pytest
from pathlib import Path
from lexsubgen.evaluations.lexsub import LexSubEvaluation
from lexsubgen import SubstituteGenerator
from lexsubgen.datasets.base_reader import DatasetReader

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


@pytest.fixture
def roberta_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "roberta.jsonnet")
    )


@pytest.fixture
def roberta_embs_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "roberta_embs.jsonnet")
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


def test_semeval_all_roberta(roberta_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/roberta.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_roberta'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_roberta'
    """
    scores = LexSubEvaluation(
        substitute_generator=roberta_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap"] == pytest.approx(50.14, 0.01), str(scores)
    assert scores["gap_normalized"] == pytest.approx(56.74, 0.02), str(scores)
    assert scores["gap_vocab_normalized"] == pytest.approx(62.22, 0.01), str(scores)
    assert scores["prec@1"] == pytest.approx(32.25), str(scores)
    assert scores["prec@3"] == pytest.approx(24.26), str(scores)
    assert scores["rec@10"] == pytest.approx(36.65), str(scores)
    assert scores["precision"] == pytest.approx(0.01), str(scores)
    assert scores["recall"] == pytest.approx(79.24), str(scores)
    assert scores["f1_score"] == pytest.approx(0.02), str(scores)


def test_coinco_roberta(roberta_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/roberta.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_roberta'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_roberta'
    """
    scores = LexSubEvaluation(
        substitute_generator=roberta_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(50.82, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(35.12), str(scores)
    assert scores["prec@3"] == pytest.approx(27.35), str(scores)
    assert scores["rec@10"] == pytest.approx(25.41), str(scores)


def test_semeval_all_roberta_embs(roberta_embs_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/roberta_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_roberta_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_roberta_embs'
    """
    scores = LexSubEvaluation(
        substitute_generator=roberta_embs_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap"] == pytest.approx(52.17, 0.01), str(scores)
    assert scores["gap_normalized"] == pytest.approx(58.74, 0.02), str(scores)
    assert scores["gap_vocab_normalized"] == pytest.approx(64.42, 0.01), str(scores)
    assert scores["prec@1"] == pytest.approx(43.19), str(scores)
    assert scores["prec@3"] == pytest.approx(31.19), str(scores)
    assert scores["rec@10"] == pytest.approx(44.61), str(scores)
    assert scores["prec@10"] == pytest.approx(16.31), str(scores)


def test_coinco_roberta_embs(roberta_embs_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/roberta_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_roberta_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_roberta_embs'
    """
    scores = LexSubEvaluation(
        substitute_generator=roberta_embs_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(54.6, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(46.54), str(scores)
    assert scores["prec@3"] == pytest.approx(36.17), str(scores)
    assert scores["rec@10"] == pytest.approx(32.1), str(scores)
