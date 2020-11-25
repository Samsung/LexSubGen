import pytest
from pathlib import Path
from lexsubgen.evaluations.lexsub import LexSubEvaluation
from lexsubgen import SubstituteGenerator
from lexsubgen.datasets.base_reader import DatasetReader

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


@pytest.fixture
def xlnet_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "xlnet.jsonnet")
    )


@pytest.fixture
def xlnet_embs_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "xlnet_embs.jsonnet")
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


def test_semeval_all_xlnet(xlnet_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/xlnet.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_xlnet'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_xlnet'
    """
    scores = LexSubEvaluation(
        substitute_generator=xlnet_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(59.12, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(31.75), str(scores)
    assert scores["prec@3"] == pytest.approx(22.83), str(scores)
    assert scores["rec@10"] == pytest.approx(34.95), str(scores)


def test_coinco_xlnet(xlnet_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/xlnet.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_xlnet'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_xlnet'
    """
    scores = LexSubEvaluation(
        substitute_generator=xlnet_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(53.39, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(38.16), str(scores)
    assert scores["prec@3"] == pytest.approx(28.58), str(scores)
    assert scores["rec@10"] == pytest.approx(26.47), str(scores)


def test_semeval_all_xlnet_embs(xlnet_embs_subst_generator, semeval_all_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/xlnet_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_xlnet_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_xlnet_embs'
    """
    scores = LexSubEvaluation(
        substitute_generator=xlnet_embs_subst_generator,
        dataset_reader=semeval_all_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(59.62, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(49.53), str(scores)
    assert scores["prec@3"] == pytest.approx(34.9), str(scores)
    assert scores["rec@10"] == pytest.approx(47.51), str(scores)


def test_coinco_xlnet_embs(xlnet_embs_subst_generator, coinco_dataset_reader):
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/xlnet_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_xlnet_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_xlnet_embs'
    """
    scores = LexSubEvaluation(
        substitute_generator=xlnet_embs_subst_generator,
        dataset_reader=coinco_dataset_reader,
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(55.63, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(51.5), str(scores)
    assert scores["prec@3"] == pytest.approx(39.92), str(scores)
    assert scores["rec@10"] == pytest.approx(35.12), str(scores)
