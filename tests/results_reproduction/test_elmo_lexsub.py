import pytest
from pathlib import Path
from lexsubgen.evaluations.lexsub import LexSubEvaluation
from lexsubgen import SubstituteGenerator
from lexsubgen.datasets.base_reader import DatasetReader

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


@pytest.fixture
def elmo_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "elmo.jsonnet")
    )


@pytest.fixture
def elmo_embs_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "elmo_embs.jsonnet")
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


def test_semeval_all_elmo(elmo_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/elmo.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_elmo'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_elmo'
    """
    scores = LexSubEvaluation(
        substitute_generator=elmo_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(53.66, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(11.58, 0.02), str(scores)
    assert scores["prec@3"] == pytest.approx(8.55, 0.02), str(scores)
    assert scores["rec@10"] == pytest.approx(13.88, 0.02), str(scores)


def test_coinco_elmo(elmo_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/elmo.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_elmo'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_elmo'
    """
    scores = LexSubEvaluation(
        substitute_generator=elmo_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(49.47, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(13.58, 0.02), str(scores)
    assert scores["prec@3"] == pytest.approx(10.86, 0.02), str(scores)
    assert scores["rec@10"] == pytest.approx(11.35, 0.02), str


def test_semeval_all_elmo_embs(elmo_embs_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/elmo_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_elmo_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_elmo_embs'
    """
    scores = LexSubEvaluation(
        substitute_generator=elmo_embs_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(54.16, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(32.0, 0.02), str(scores)
    assert scores["prec@3"] == pytest.approx(22.2, 0.02), str(scores)
    assert scores["rec@10"] == pytest.approx(31.82, 0.02), str(scores)


def test_coinco_elmo_embs(elmo_embs_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/elmo_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_elmo_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_elmo_embs'
    """
    scores = LexSubEvaluation(
        substitute_generator=elmo_embs_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(52.22, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(35.96, 0.02), str(scores)
    assert scores["prec@3"] == pytest.approx(26.62, 0.02), str(scores)
    assert scores["rec@10"] == pytest.approx(23.8, 0.02), str(scores)
