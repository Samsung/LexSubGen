import pytest
from pathlib import Path
from lexsubgen.evaluations.lexsub import LexSubEvaluation
from lexsubgen import SubstituteGenerator
from lexsubgen.datasets.base_reader import DatasetReader

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


def test_semeval_all_xlnet():
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/semeval_xlnet.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_xlnet'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_xlnet'
    """
    scores = LexSubEvaluation.from_configs(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "semeval_xlnet.jsonnet"),
        str(CONFIGS_PATH / "dataset_readers" / "lexsub" / "semeval_all.jsonnet"),
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(59.11, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(31.7), str(scores)
    assert scores["prec@3"] == pytest.approx(22.8), str(scores)
    assert scores["rec@10"] == pytest.approx(34.93), str(scores)


def test_coinco_xlnet():
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/coinco_xlnet.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_xlnet'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_xlnet'
    """
    scores = LexSubEvaluation.from_configs(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "coinco_xlnet.jsonnet"),
        str(CONFIGS_PATH / "dataset_readers" / "lexsub" / "coinco.jsonnet"),
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(53.39, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(38.16), str(scores)
    assert scores["prec@3"] == pytest.approx(28.58), str(scores)
    assert scores["rec@10"] == pytest.approx(26.47), str(scores)


def test_semeval_all_xlnet_embs():
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/semeval_xlnet_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/semeval_all.jsonnet
        --run-dir='debug/lexsub-all-models/semeval_all_xlnet_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='semeval_all_xlnet_embs'
    """
    scores = LexSubEvaluation.from_configs(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "semeval_xlnet_embs.jsonnet"),
        str(CONFIGS_PATH / "dataset_readers" / "lexsub" / "semeval_all.jsonnet"),
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(59.62, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(49.53), str(scores)
    assert scores["prec@3"] == pytest.approx(34.88), str(scores)
    assert scores["rec@10"] == pytest.approx(47.47), str(scores)


def test_coinco_xlnet_embs():
    """
    Reproduction command:
    python lexsubgen/evaluations/lexsub.py solve
        --substgen-config-path configs/subst_generators/lexsub/coinco_xlnet_embs.jsonnet
        --dataset-config-path configs/dataset_readers/lexsub/coinco.jsonnet
        --run-dir='debug/lexsub-all-models/coinco_xlnet_embs'
        --force
        --experiment-name='lexsub-all-models'
        --run-name='coinco_xlnet_embs'
    """
    scores = LexSubEvaluation.from_configs(
        str(CONFIGS_PATH / "subst_generators" / "lexsub" / "coinco_xlnet_embs.jsonnet"),
        str(CONFIGS_PATH / "dataset_readers" / "lexsub" / "coinco.jsonnet"),
    ).evaluate()["mean_metrics"]
    assert scores["gap_normalized"] == pytest.approx(55.63, 0.02), str(scores)
    assert scores["prec@1"] == pytest.approx(51.5), str(scores)
    assert scores["prec@3"] == pytest.approx(39.92), str(scores)
    assert scores["rec@10"] == pytest.approx(35.12), str(scores)
