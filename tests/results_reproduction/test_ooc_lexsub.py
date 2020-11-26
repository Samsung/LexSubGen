import pytest
from pathlib import Path
from lexsubgen.evaluations.lexsub import LexSubEvaluation
from lexsubgen import SubstituteGenerator
from lexsubgen.datasets.base_reader import DatasetReader

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


@pytest.fixture
def ooc_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "ooc.jsonnet")
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


def test_semeval_all_ooc(ooc_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/ooc.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_ooc'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_ooc'
    """
    scores = LexSubEvaluation(
        substitute_generator=ooc_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(44.65, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(16.82), str(scores)
    assert scores["prec@3"] == pytest.approx(12.83), str(scores)
    assert scores["rec@10"] == pytest.approx(18.36), str(scores)


def test_coinco_ooc(ooc_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/ooc.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_ooc'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_ooc'
    """
    scores = LexSubEvaluation(
        substitute_generator=ooc_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(46.3, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(19.58), str(scores)
    assert scores["prec@3"] == pytest.approx(15.03), str(scores)
    assert scores["rec@10"] == pytest.approx(12.99), str(scores)
