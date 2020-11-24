import logging
import os
import shutil
import unicodedata
from pathlib import Path
from typing import List

import wget

from lexsubgen.utils.file import extract_archive

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)

LEXSUB_DATASETS_URL = "https://github.com/stephenroller/naacl2016/archive/master.zip"


def download_dataset(url: str, dataset_path: str):
    """
    Method for downloading dataset from a given URL link.
    After download dataset will be saved in the dataset_path directory.

    Args:
        url: URL link to dataset.
        dataset_path: Directory path to save the downloaded dataset.

    Returns:

    """
    os.makedirs(dataset_path, exist_ok=True)
    logger.info(f"Downloading file from '{url}'...")
    filename = wget.download(url, out=str(dataset_path))
    logger.info(f"File {filename} is downloaded to '{dataset_path}'.")
    filename = Path(filename)

    # Extract archive if needed
    extract_archive(arch_path=filename, dest=dataset_path)

    # Delete archive
    if os.path.isfile(filename):
        os.remove(filename)
    elif os.path.isdir(filename):
        shutil.rmtree(filename)


def strip_accents(s: str) -> str:
    """
    Remove accents from given string:
    Example: strip_accents("Málaga") -> Malaga

    Args:
        s: str - string to process
    Returns:
        string without accents
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def split_line(line: str, sep: str = " ") -> List[str]:
    """
    Method for splitting line by given separator 'sep'.

    Args:
        line: Input line to split.
        sep: Separator char.
    Returns:
        line: List of parts of the input line.
    """
    line = [part.strip() for part in line.split(sep=sep)]
    return line
