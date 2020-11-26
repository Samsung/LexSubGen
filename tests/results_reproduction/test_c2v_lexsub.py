import pytest
from pathlib import Path
from lexsubgen.evaluations.lexsub import LexSubEvaluation
from lexsubgen import SubstituteGenerator
from lexsubgen.datasets.base_reader import DatasetReader

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


@pytest.fixture
def c2v_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "c2v.jsonnet")
    )


@pytest.fixture
def c2v_embs_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "c2v_embs.jsonnet")
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


def test_semeval_all_c2v(c2v_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/c2v.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_c2v'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_c2v'
    """
    scores = LexSubEvaluation(
        substitute_generator=c2v_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(55.82, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(7.79), str(scores)
    assert scores["prec@3"] == pytest.approx(5.92), str(scores)
    assert scores["rec@10"] == pytest.approx(11.03), str(scores)


def test_coinco_c2v(c2v_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/c2v.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_c2v'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_c2v'
    """
    scores = LexSubEvaluation(
        substitute_generator=c2v_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(48.32, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(8.01), str(scores)
    assert scores["prec@3"] == pytest.approx(6.63), str(scores)
    assert scores["rec@10"] == pytest.approx(7.54), str(scores)


def test_semeval_all_c2v_embs(c2v_embs_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/c2v_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_c2v_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_c2v_embs'
    """
    scores = LexSubEvaluation(
        substitute_generator=c2v_embs_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(53.39, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(28.01), str(scores)
    assert scores["prec@3"] == pytest.approx(21.72), str(scores)
    assert scores["rec@10"] == pytest.approx(33.52), str(scores)


def test_coinco_c2v_embs(c2v_embs_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/c2v_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_c2v_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_c2v_embs'
    """
    scores = LexSubEvaluation(
        substitute_generator=c2v_embs_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(50.73, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(29.64), str(scores)
    assert scores["prec@3"] == pytest.approx(24.0), str(scores)
    assert scores["rec@10"] == pytest.approx(21.97), str(scores)
