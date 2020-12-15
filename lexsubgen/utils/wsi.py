import os
from pathlib import Path

from lexsubgen.datasets.utils import download_dataset
from lexsubgen.utils.register import CACHE_DIR

SEMEVAL2013URL = "https://www.cs.york.ac.uk/semeval-2013/task13/data/uploads/semeval-2013-task-13-test-data.zip"
SEMEVAL2010URL = "https://www.cs.york.ac.uk/semeval2010_WSI/files/evaluation.zip"
SEMEVAL2010TRAINURL = "https://www.cs.york.ac.uk/semeval2010_WSI/files/training_data.tar.gz"
SEMEVAL2010TESTURL = "https://www.cs.york.ac.uk/semeval2010_WSI/files/test_data.tar.gz"


def download_semeval_2013_data_if_not_exists() -> str:
    """
    Donwloads semeval 2013 task 13 dataset if it's not in cache directory.
    """
    dataset_path = Path(CACHE_DIR) / "wsi" / "semeval-2013"
    if not os.path.exists(dataset_path):
        download_dataset(SEMEVAL2013URL, dataset_path)
    return str(dataset_path / "SemEval-2013-Task-13-test-data")


def download_semeval_2010_data_if_not_exists() -> str:
    dataset_path = Path(CACHE_DIR) / "wsi" / "semeval-2010"
    if not os.path.exists(dataset_path):
        download_dataset(SEMEVAL2010URL, dataset_path)
        download_dataset(SEMEVAL2010TRAINURL, dataset_path)
        download_dataset(SEMEVAL2010TESTURL, dataset_path)
    return str(dataset_path)
