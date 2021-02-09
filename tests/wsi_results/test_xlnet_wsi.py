import pytest
from pathlib import Path
from lexsubgen.evaluations.wsi import WSIEvaluation
from lexsubgen import SubstituteGenerator
from lexsubgen.datasets.base_reader import DatasetReader
from lexsubgen.clusterizers.agglo import SubstituteClusterizer

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


@pytest.fixture
def clusterizer():
    return SubstituteClusterizer.from_config(
        str(CONFIGS_PATH / "clusterizers" / "agglo.jsonnet")
    )


@pytest.fixture
def xlnet_subst_generator():
    return SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "wsi" / "xlnet.jsonnet")
    )


@pytest.fixture
def semeval_2010_dataset_reader():
    return DatasetReader.from_config(
        str(CONFIGS_PATH / "dataset_readers" / "wsi" / "semeval_2010.jsonnet")
    )


@pytest.fixture
def semeval_2013_dataset_reader():
    return DatasetReader.from_config(
        str(CONFIGS_PATH / "dataset_readers" / "wsi" / "semeval_2013.jsonnet")
    )


def test_semeval_2013_xlnet(xlnet_subst_generator, semeval_2013_dataset_reader, clusterizer):
    """
    Reproduction command:
    python lexsubgen/evaluations/wsi.py solve
        --dataset-config-path configs/dataset_readers/wsi/semeval_2013.jsonnet
        --substgen-config-path configs/subst_generators/wsi/xlnet.jsonnet
        --clusterizer-config-path configs/clusterizers/agglo.jsonnet
        --run-dir debug/xlnet-semeval-2013 --force
        --experiment-name='wsi'
        --run-name='xlnet-semeval-2013'
        --use-pos-tags=False
    """
    scores = WSIEvaluation(
        substitute_generator=xlnet_subst_generator,
        dataset_reader=semeval_2013_dataset_reader,
        clusterizer=clusterizer,
        use_pos_tags=False,
    ).evaluate()["mean_metrics"]
    assert scores["S13_AVG"] == pytest.approx(33.4114, 0.001), str(scores)


def test_semeval_2010_xlnet(xlnet_subst_generator, semeval_2010_dataset_reader, clusterizer):
    """
    Reproduction command:
    python lexsubgen/evaluations/wsi.py solve
        --dataset-config-path configs/dataset_readers/wsi/semeval_2010.jsonnet
        --substgen-config-path configs/subst_generators/wsi/xlnet.jsonnet
        --clusterizer-config-path configs/clusterizers/agglo.jsonnet
        --run-dir debug/xlnet-semeval-2010 --force
        --experiment-name='wsi'
        --run-name='xlnet-semeval-2010'
        --use-pos-tags=False
        --verbose
        --save-instance-results
    """
    scores = WSIEvaluation(
        substitute_generator=xlnet_subst_generator,
        dataset_reader=semeval_2010_dataset_reader,
        clusterizer=clusterizer,
        use_pos_tags=False,
    ).evaluate()["mean_metrics"]
    assert scores["S10_AVG"] == pytest.approx(52.1804, 0.001), str(scores)


def test_semeval_2013_xlnet_embs(semeval_2013_dataset_reader, clusterizer):
    """
    Reproduction command:
    python lexsubgen/evaluations/wsi.py solve
        --dataset-config-path configs/dataset_readers/wsi/semeval_2013.jsonnet
        --substgen-config-path configs/subst_generators/wsi/xlnet_embs_se13.jsonnet
        --clusterizer-config-path configs/clusterizers/agglo.jsonnet
        --run-dir debug/xlnet-embs-semeval-2013 --force
        --experiment-name='wsi'
        --run-name='xlnet-embs-semeval-2013'
        --use-pos-tags=False
        --verbose
        --save-instance-results
    """

    xlnet_embs_subst_generator = SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "wsi" / "xlnet_embs_se13.jsonnet")
    )
    scores = WSIEvaluation(
        substitute_generator=xlnet_embs_subst_generator,
        dataset_reader=semeval_2013_dataset_reader,
        clusterizer=clusterizer,
        use_pos_tags=False,
    ).evaluate()["mean_metrics"]
    assert scores["S13_AVG"] == pytest.approx(37.3546, 0.001), str(scores)


def test_semeval_2010_xlnet_embs(semeval_2010_dataset_reader, clusterizer):
    """
    Reproduction command:
    python lexsubgen/evaluations/wsi.py solve
        --dataset-config-path configs/dataset_readers/wsi/semeval_2010.jsonnet
        --substgen-config-path configs/subst_generators/wsi/xlnet_embs_se10.jsonnet
        --clusterizer-config-path configs/clusterizers/agglo.jsonnet
        --run-dir debug/xlnet-embs-semeval-2010 --force
        --experiment-name='wsi'
        --run-name='xlnet-embs-semeval-2010'
        --use-pos-tags=False
        --verbose
        --save-instance-results
    """
    xlnet_embs_subst_generator = SubstituteGenerator.from_config(
        str(CONFIGS_PATH / "subst_generators" / "wsi" / "xlnet_embs_se10.jsonnet")
    )
    scores = WSIEvaluation(
        substitute_generator=xlnet_embs_subst_generator,
        dataset_reader=semeval_2010_dataset_reader,
        clusterizer=clusterizer,
        use_pos_tags=False,
    ).evaluate()["mean_metrics"]
    assert scores["S10_AVG"] == pytest.approx(54.1969, 0.001), str(scores)
