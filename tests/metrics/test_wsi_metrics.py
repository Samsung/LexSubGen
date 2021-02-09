import pytest
from pathlib import Path
from lexsubgen.utils.register import CACHE_DIR

from lexsubgen.metrics.wsi_metrics import _compute_semeval_2010_metrics


def test_compute_semeval_2010_metrics():
    data_path = Path(CACHE_DIR) / "wsi" / "semeval-2010"
    gold_labels_path = data_path / "evaluation" / "unsup_eval" / "keys" / "all.key"
    fscore = _compute_semeval_2010_metrics(
        data_path / "evaluation" / "unsup_eval" / "fscore.jar",
        gold_labels_path,
        gold_labels_path,
    )
    vmeasure = _compute_semeval_2010_metrics(
        data_path / "evaluation" / "unsup_eval" / "vmeasure.jar",
        gold_labels_path,
        gold_labels_path,
    )
    assert len(fscore) == 101, str(fscore)
    assert len(vmeasure) == 101, str(vmeasure)
    for word, (fs, prec, rec) in fscore.items():
        assert fs == pytest.approx(100.0)
        if word != "all":
            assert prec == pytest.approx(100.0)
            assert rec == pytest.approx(100.0)

    for word, (vm, homogeneity, completeness) in vmeasure.items():
        assert vm == pytest.approx(100.0)
        if word != "all":
            assert homogeneity == pytest.approx(100.0)
            assert completeness == pytest.approx(100.0)
