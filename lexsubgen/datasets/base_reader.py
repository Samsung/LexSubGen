import logging
from pathlib import Path
from typing import List, Union, Optional

from lexsubgen.datasets.utils import strip_accents, download_dataset
from lexsubgen.utils.params import build_from_config_path
from lexsubgen.utils.register import DATASETS_DIR

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)


class DatasetReader:
    def __init__(
        self,
        dataset_name: str,
        data_root_path: Union[str, Path] = DATASETS_DIR,
        url: Optional[str] = None,
    ):
        """
        Abstract class of dataset reader.
        It must provide interfaces for following use cases:
        1. Downloading dataset and putting it on the cache directory.
        2. Read dataset by its name.

        Args:
            dataset_name: Alias for dataset naming.
            data_root_path: Path for all available datasets.
                Datasets will be downloaded to this directory.
            url: Link for downloading dataset.
        """
        self.dataset_path = Path(data_root_path) / dataset_name
        self.url = url
        if not self.dataset_path.exists():
            download_dataset(self.url, self.dataset_path)

    @classmethod
    def from_config(cls, config_path):
        """
        Method for creating dataset reader instance of 'cls' class.

        Args:
            config_path: Path to file with dataset reader config.
        Returns:
            dataset_reader: Instance of class 'cls'
        """
        dataset_reader, _ = build_from_config_path(config_path)
        return dataset_reader

    def read_dataset(self):
        """
        Abstract method for reading dataset.
        It must be overridden in inherited classes.

        Returns:
            It must be specified in inherited classes.
        """
        raise NotImplementedError("Override this method in subclass")

    def read_file(
        self, file_path: Union[str, Path], accents: bool = False, lower: bool = False
    ) -> List[str]:
        file_path = Path(file_path)
        if not file_path.exists():
            if self.url is None:
                raise FileNotFoundError(f"File {file_path} doesn't exist!")
            download_dataset(self.url, self.dataset_path)

        logger.info(msg=f"Reading data from {file_path} file...")
        with file_path.open("r") as f:
            data = f.readlines()

        while "\n" in data:
            data.remove("\n")
        if accents:
            data = [strip_accents(line) for line in data]
        if lower:
            data = [line.lower() for line in data]
        logger.info(msg=f"Done. File contains {len(data)} lines")
        return data