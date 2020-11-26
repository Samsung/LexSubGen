import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import wget
from context2vec.common.model_reader import ModelReader
from overrides import overrides

from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator
from lexsubgen.utils.file import extract_archive, download_large_gdrive_file
from lexsubgen.utils.register import CACHE_DIR

CONTEXT2VEC_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "c2v_ukwac": {
        "wget": "http://irsrv2.cs.biu.ac.il/downloads/context2vec/context2vec.ukwac.model.package.tar.gz",
        # This is a copy of the file above because the link doesn't work anymore.
        "gdrive_id": "1OxeWZl3bdQnM5AMTye_FQRATnvsNv1PY",
    },
    "c2v_mscc": {
        "wget": "http://irsrv2.cs.biu.ac.il/downloads/context2vec/context2vec.mscc.model.package.tar.gz"
    },
}


logger = logging.getLogger(Path(__file__).name)


class Context2VecProbEstimator(EmbSimProbEstimator):
    def __init__(
        self,
        model_name: str = "c2v_ukwac",
        embedding_similarity: bool = False,
        temperature: float = 1.0,
        sim_func: str = "dot-product",
        verbose: bool = False,
    ):
        """
        Probability esetimator based on context2vec model.
        See Oren Melamud et al. "context2vec: Learning Generic Context Embedding with Bidirectional LSTM"

        Args:
            model_name: name of the pre-trained context2vec model. It could be a path to the model parameters
            or a name of the pre-trained model: c2v_ukwac or c2v_mscc.
            embedding_similarity: whether to use word embedding similarity to get probability distribution.
            temperature: value of the temperature
            sim_func: similarity function to be used, currently supported dot-product, cosine, euclidean
            verbose: whether to print misc output
        """
        # TODO: mapping to c2v versions from model_name
        super(Context2VecProbEstimator, self).__init__(
            model_name=model_name,
            temperature=temperature,
            sim_func=sim_func,
            verbose=verbose,
        )
        self.model_name = model_name
        self.embedding_similarity = embedding_similarity

        self.descriptor = {
            "Prob_generator": {
                "name": "context2vec",
                "model_name": model_name,
                "temperature": self.temperature,
                "emb_similarity": self.embedding_similarity,
            }
        }

        self.register_model()
        self.logger.debug("Probability estimator created.")
        self.logger.debug(f"Config:\n{json.dumps(self.descriptor, indent=4)}")

    def register_model(self):
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        if self.model_name not in Context2VecProbEstimator.loaded.keys():
            model_reader = ModelReader(self.get_model_params_path(self.model_name))
            Context2VecProbEstimator.loaded[self.model_name] = {
                "model": model_reader.model.context2vec,
                "word2id": model_reader.word2index,
                "embeddings": model_reader.w,
            }
            Context2VecProbEstimator.loaded[self.model_name]["ref_count"] = 1
        else:
            Context2VecProbEstimator.loaded[self.model_name]["ref_count"] += 1

    @overrides
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int],
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        Process all input data with batches.

        Args:
            tokens_lists: list of tokenized sequences,  each list corresponds to one tokenized example.
            target_ids: indices of target words from all tokens lists.
                E.g.:
                token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                This means that we want to get probability distribution for words "world" and "stackoverflow".

        Returns:
            `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.
        """
        if self.embedding_similarity:
            logits = self.get_emb_similarity(tokens_lists, target_ids)
        else:
            logits = self.predict(tokens_lists, target_ids)
        return logits, self.word2id

    def predict(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> np.ndarray:
        """
        Get log probability distribution over vocabulary.

        Args:
            tokens_lists: list of contexts
            target_ids: target word indexes

        Returns:
            `numpy.ndarray`, matrix with rows - log-prob distribution over vocabulary.
        """
        batch_of_logits = []
        for sample_idx, (tokens, target_idx) in enumerate(
            zip(tokens_lists, target_ids)
        ):
            tokens = [token.lower() for token in tokens]
            context_vector = self.model(tokens, target_idx)
            context_vector = context_vector / np.sqrt(
                context_vector.dot(context_vector)
            )
            logits = self.embeddings.dot(context_vector)
            batch_of_logits.append(logits)
        logits = np.vstack(batch_of_logits)
        return logits

    @staticmethod
    def get_model_params_path(model_name_or_path: str) -> str:
        """
        Acquires context2vec model parameters path.

        Args:
            model_name_or_path: model name or path to the parameters

        Returns:
            path to model parameters
        """
        if model_name_or_path in CONTEXT2VEC_PRETRAINED_MODEL_ARCHIVE_MAP:
            return load_c2v_model(model_name_or_path)
        return model_name_or_path


def find_parameters_path(model_path: Path) -> Optional[str]:
    """
    Searches for context2vec model parameters in a given model folder.

    Args:
        model_path: path to the directory with model files.

    Returns:
        path to the model parameters file
    """
    for file_ in model_path.iterdir():
        if not file_.is_file():
            continue
        if file_.suffix == ".params":
            return str(file_.absolute())
    return


def load_c2v_model(model_name: str) -> str:
    """
    Loads pre-trained context2vec model if there is no one.

    Args:
        model_name: name of the pre-trained model.

    Returns:
        path to the model parameters file.
    """
    if model_name not in CONTEXT2VEC_PRETRAINED_MODEL_ARCHIVE_MAP:
        raise ValueError(f"There is no pre-trained c2v model named: {model_name}.")
    model_path = CACHE_DIR / "resources" / "c2v" / model_name
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=False)

    params_path = find_parameters_path(model_path)

    if params_path is None:
        logger.info("Loading context2vec params...")
        if "gdrive_id" not in CONTEXT2VEC_PRETRAINED_MODEL_ARCHIVE_MAP[model_name]:
            model_url = CONTEXT2VEC_PRETRAINED_MODEL_ARCHIVE_MAP[model_name]["wget"]
            archive = wget.download(model_url, out=str(model_path))
        else:
            archive = f"{model_path}.tar.gz"
            gdrive_id = CONTEXT2VEC_PRETRAINED_MODEL_ARCHIVE_MAP[model_name]["gdrive_id"]
            download_large_gdrive_file(gdrive_id, archive)
        extract_archive(arch_path=archive, dest=str(model_path))
        logger.info("Params loaded.")

        params_path = find_parameters_path(model_path)
        Path(archive).unlink()
    return params_path
