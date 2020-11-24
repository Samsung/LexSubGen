from pathlib import Path
from typing import Union, Optional

import pandas as pd
from overrides import overrides

from .base_reader import DatasetReader

# TODO: Add URL link to SNIPS dataset
NLU_DATASET_DRIVE_URLS = {"snips": ""}
# List of supported data sets.
NLU_DATASETS = ("snips_train", "snips_valid", "snips_test")


class NLUDatasetReader(DatasetReader):
    DATA_COLUMNS = ["utterance", "intent", "slots"]

    def __init__(
        self,
        dataset_name: str,
        data_root_path: Union[str, Path] = DATASETS_DIR,
        url: Optional[str] = None,
    ):
        """
        Reader for NLU datasets.
        Args:
            dataset_name: Alias for dataset naming.
            data_root_path: Path for all available datasets. Datasets will be downloaded to this directory.
            url: Link for downloading dataset.
        """
        assert (
            "_" in dataset_name
        ), f"Invalid dataset name ({dataset_name}). Available datasets: {NLU_DATASETS}"
        _dataset_name, _dataset_part = dataset_name.split("_")
        if url is None and dataset_name in NLU_DATASETS:
            # dataset from list of supported nlu data sets
            url = NLU_DATASET_DRIVE_URLS[_dataset_name]
        dataset_name = _dataset_name + "/" + _dataset_part
        super(NLUDatasetReader, self).__init__(
            dataset_name=dataset_name, data_root_path=data_root_path, url=url
        )

    @overrides
    def read_dataset(self) -> pd.DataFrame:
        """
        NLU dataset consists of 3 different files:
            1. seq.in - file with raw utterances.
            2. seq.out - file with Slot tags in BIO format.
            3. label - file with Intent labels.

        Returns:
            df: pandas DataFrame that contains following columns:
                'utterance', "intent', 'slots'
        """
        utterances = self.read_file(self.dataset_path / "seq.in")
        slots = self.read_file(self.dataset_path / "seq.out")
        intents = self.read_file(self.dataset_path / "label")

        dataset = {column: [] for column in self.DATA_COLUMNS}
        for utter, slot, intent in zip(utterances, slots, intents):
            dataset["utterance"].append(utter)
            dataset["slots"].append(slot)
            dataset["intent"].append(intent)
        df = pd.DataFrame(data=dataset)
        return df
