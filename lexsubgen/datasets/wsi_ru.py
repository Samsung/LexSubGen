import csv
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import nltk
import pandas as pd

from lexsubgen.datasets.wsi import wsi_logging_info
from lexsubgen.utils.register import CACHE_DIR

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)


def download_russe_datasets(download_path: Path):
    if os.path.exists(str(download_path)):
        return
    logger.info("Downloading russe-wsi datasets from https://github.com/nlpub/russe-wsi-kit.git")
    # TODO: use subprocess
    os.system(f'git clone https://github.com/nlpub/russe-wsi-kit.git {download_path / "russe-wsi-kit"}')
    os.system(f'mv {download_path / "russe-wsi-kit" / "data"}/* {download_path}')
    os.system(f'rm -rf {download_path / "russe-wsi-kit"}')
    logger.info("Downloading done.")


def copy_russe_datasets(datapath: str, download_path: Path):
    if os.path.exists(str(download_path / "main")):
        return
    os.system(f"cp -rf {datapath}/* {download_path}")


class RusseBTSRNCDatasetReader:
    data_root_path = Path(CACHE_DIR) / "wsi" / "russe"
    dataset_name = "bts-rnc"
    inner_path = Path("main") / "bts-rnc"
    # gold_labels_path = Path(dataset_name) / "evaluation" / "unsup_eval" / "keys" / "all.key"
    df_columns = ["context_id", "group_by", "target_lemma", "pos_tag", "sentence", "target_id"]

    def __init__(
        self,
        part: str = "train",
        datapath: str = None,
        tokenize: bool = True
    ):
        """
        Reader for RUSSE WSI dataset - bts-rnc: https://github.com/nlpub/russe-wsi-kit
        Args:
            part: part of the bts-rnc dataset
        """
        self.part = part
        self.tokenize = tokenize
        self.inner_path = self.inner_path / f"{self.part}.csv"
        if not self.data_root_path.exists() and datapath is None:
            raise RuntimeError(
                f"Dataset from repository 'https://github.com/nlpub/russe-wsi-kit' has a lot of bugs. "
                f"So you should specify @datapath argument in the config file."
            )
        self.data_root_path.mkdir(parents=True, exist_ok=True)
        copy_russe_datasets(datapath, self.data_root_path)
        # download_russe_datasets(self.data_root_path)

    @wsi_logging_info(logger)
    def read_dataset(self, limit: int = None) -> Tuple[pd.DataFrame, Dict[str, str], Any]:
        """
        Reads defined part of bts-rnc dataset
        Returns:
            `pandas.DataFrame`: dataframe that contains following columns:
                'context_id', 'group_by', 'target_lemma', 'pos_tag', 'sentence', 'target_id'
            dict: for each context_id contains the gold label
        """

        df = pd.read_csv(
            self.data_root_path / self.inner_path,
            sep="\t", encoding="utf-8",nrows=limit,
            quoting=csv.QUOTE_MINIMAL
        )

        df.rename(columns={"word": "target_lemma"}, inplace=True)

        df["group_by"] = df["target_lemma"]
        df["pos_tag"] = None

        target_ids = []
        sentences = []
        for ctx, pos, target_word in zip(
            df["context"], df["positions"], df["target_lemma"]
        ):
            first_appearance = pos.split(",")[0]
            posl, posr = (int(p) for p in first_appearance.split("-"))

            if self.tokenize:
                tokenized_ctx = nltk.word_tokenize(ctx)
                word = ctx[posl:posr+1] # if dataset files weren't fixed use word = ctx[posl:posr+1]
                word_indexes = [
                    token_index
                    for token_index, token in enumerate(tokenized_ctx)
                    if word in token
                ]
                # first appearance of the target word
                target_ids.append(word_indexes[0])
                sentences.append(tokenized_ctx)
            else:
                sentences.append([ctx[:posl], word, ctx[posr+1:]]) # end of word at posr index including Thus we took right context from the next char after word 
                target_ids.append(1)

        df["sentence"] = sentences
        df["target_id"] = target_ids
        df["context_id"] = df["context_id"].astype(str)

        gold_labels = {
            idx: label for label, idx in zip(df.gold_sense_id, df.context_id)
        }
        return df[self.df_columns], gold_labels, None
