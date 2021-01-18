import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import h5py
import numpy as np
import torch
import wget
from allennlp.commands.elmo import ElmoEmbedder

from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator
from lexsubgen.utils.dists import fast_np_sparse_batch_combine_two_dists
from lexsubgen.utils.register import CACHE_DIR
RESOURCES_DIR = Path(__file__).resolve().parent.parent.parent / "resources"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

ELMO_HIDDEN_SIZE = 512
ELMO_CNN_LAYER_NUM = 0
ELMO_LSTM_LAYER_NUMS = [1, 2]
ELMO_DIRECTIONS = ('forward', 'backward', 'both')

ELMO_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'elmo-en': {
        'copy': False,
        'options_path': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
                        "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        'weights_path': "http://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
                        "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
        'softmax_weights_path': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
                                "2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_softmax_weights.hdf5",
        'vocab_path': "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/"
                      "2x4096_512_2048cnn_2xhighway/vocab-2016-09-10.txt"
    },
    'elmo-ru-news': {
        'copy': True,
        'options_path': Path("options.json"),
        'weights_path': Path("dumps") / "weights_epoch_n_0.hdf5",
        'softmax_weights_path': Path("dumps") / "weights_epoch_n_0.hdf5",
        'vocab_path': Path("vocab.txt")
    }
}

WARM_UP_SENT_EN = "A computer is a machine that can be instructed to carry out sequences" \
                  "of arithmetic or logical operations automatically via computer programming." \
                  "Modern computers have the ability to follow generalized sets of operations," \
                  "called programs. These programs enable computers to perform an extremely" \
                  "wide range of tasks. A \"complete\" computer including the hardware, the" \
                  "operating system (main software), and peripheral equipment required and" \
                  "used for \"full\" operation can be referred to as a computer system. This" \
                  "term may as well be used for a group of computers that are connected and" \
                  "work together, in particular a computer network or computer cluster.".split()

WARM_UP_SENT_RU = "История всех городов и населённых пунктов начинается с их основания , с момента появления " \
                  "на их территории первых людей и до окончания беспрерывного проживания . По традиции историю " \
                  "населённых пунктов нередко ведут с момента их первого упоминания в письменных источниках ( " \
                  "других способов передачи более достоверной информации пока нет ) . За период существования " \
                  "города может неоднократно меняться статус и его название . Сегодня известны тысячи реальных " \
                  "древних и средневековых городов , открытых археологами . На территории некоторых построены " \
                  "новые города , а в некоторых продолжается непрерывность жизни . Но история — это повествование " \
                  "о том , что узнано , исследовано . Поэтому историю городов в основном ведут от первого " \
                  "письменного упоминания .".split()

WARM_UP_DICT = {"elmo-en": WARM_UP_SENT_EN, "elmo-ru-news": WARM_UP_SENT_RU}

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)


class ElmoProbEstimator(EmbSimProbEstimator):
    def __init__(
        self,
        model_name: str = "elmo-en",
        weights_path: str = None, 
        cutoff_vocab: Optional[int] = None,
        add_bias: bool = False,
        embedding_similarity: bool = False,
        direction: str = "both",
        temperature: float = 1.0,
        cuda_device: int = -1,
        sim_func: str = "dot-product",
        verbose: bool = False,
    ):
        """
        Probability estimator based on ELMo model. See M.E. Peters "Deep contextualized word representations"
        for more details on the underlying model.

        Args:
            model_name: name of the model or path to the folder containing it
            cutoff_vocab: how many words cut from vocabulary (if None vocabulary doesn't change)
            add_bias: boolean flag for option "Add bias"
                If add_bias = True, we add bias vector after matrix multiplication by softmax weights
            embedding_similarity: whether to compute ELMo word embedding similarity instead of the full model
            direction: in which direction to process context: forward, backward or both
            temperature: temperature by which to divide log-probs
            cuda_device: CUDA device to load model to
            sim_func: name of similarity function to use in order to compute embedding similarity
            verbose: whether to print misc information
        """
        super(ElmoProbEstimator, self).__init__(
            model_name=model_name,
            verbose=verbose,
            sim_func=sim_func,
            temperature=temperature,
        )
        self.cutoff_vocab = cutoff_vocab
        self.add_bias = add_bias

        if direction not in ELMO_DIRECTIONS:
            raise ValueError(f"Wrong direction. Choose one from {ELMO_DIRECTIONS}")
        self.direction = direction

        if cuda_device >= 0 and not torch.cuda.is_available():
            logger.info(f"Cuda device '{cuda_device}' isn't available, "
                        f"so it is set to -1")
            cuda_device = -1
        self.cuda_device = cuda_device

        self.embedding_similarity = embedding_similarity
        # Modifying batch_size may lead to different results
        self.batch_size = 100
        self.weights_path = weights_path
        self.loaded_name = f"{model_name}#{cutoff_vocab}"

        # This function will load vocabularies, weights and  warm up model
        self.register_model()

    @staticmethod
    def warm_up_elmo(
        elmo: ElmoEmbedder,
        model_name: str,
        batch_size: int,
        verbose: bool = False
    ) -> None:
        """
        Pass a few sentences to Elmo for accumulate internal state.
        Internal state controls by AllenNLP lib.

        Args:
            elmo: `allennlp.ElmoEmbedder` model
            model_name: model name
            batch_size: size of the batch
            verbose: whether to print misc information
        """
        if verbose:
            logger.info("Loading ELMo model...")
            logger.info("Warming up ELMo...")

        warm_up_sentence = WARM_UP_DICT[model_name]
        # Running a few sentences in ELMo will set it to a better state than initial zeros
        _ = list(elmo.embed_sentences([warm_up_sentence] * batch_size, batch_size))

        if verbose:
            logger.info("Warming up done!")

    def register_model(self):
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        (
            options_path,
            weights_path,
            softmax_weights_path,
            vocab_path,
        ) = self.get_model_part_paths(self.model_name)

        if self.loaded_name not in ElmoProbEstimator.loaded:
            # Creation of instance ElmoEmbedder from AllenNLP library
            # load weights if needed
            elmo = ElmoEmbedder(
                options_file=str(options_path),
                weight_file=str(weights_path),
                cuda_device=self.cuda_device
            )

            ElmoProbEstimator.warm_up_elmo(elmo, self.model_name, self.batch_size, self.verbose)

            with h5py.File(softmax_weights_path, "r") as f:
                if self.cutoff_vocab is not None:
                    elmo_softmax_w = f["softmax/W"][: self.cutoff_vocab, :].transpose()
                    elmo_softmax_b = f["softmax/b"][: self.cutoff_vocab]
                else:
                    elmo_softmax_w = f["softmax/W"][:, :].transpose()
                    elmo_softmax_b = f["softmax/b"][:]

            # Loading ELMo vocabularies (original and lemmatized)
            word_forms_vocab, remove_ids = ElmoProbEstimator.load_vocab(
                str(vocab_path), self.cutoff_vocab, self.verbose
            )

            # Removing
            elmo_softmax_w = np.delete(elmo_softmax_w, remove_ids, axis=1)
            elmo_softmax_b = np.delete(elmo_softmax_b, remove_ids)
            word2id = {w: i for i, w in enumerate(word_forms_vocab)}

            ElmoProbEstimator.loaded[self.loaded_name] = {
                "model": elmo,
                "elmo_softmax_w": elmo_softmax_w,
                "elmo_softmax_b": elmo_softmax_b,
                "word2id": word2id,
            }
            ElmoProbEstimator.loaded[self.loaded_name]["ref_count"] = 1
        else:
            ElmoProbEstimator.loaded[self.loaded_name]["ref_count"] += 1
        # compatibility with other models naming
        self.model_name = self.loaded_name

    @property
    def embeddings(self):
        """
        ELMo embedding matrix before softmax

        Returns:
            `numpy.ndarray`, embedding matrix weights
        """
        return self.loaded[self.model_name]["elmo_softmax_w"].T

    @property
    def bias(self):
        """
        ELMo bias term before softmax.

        Returns:
            `numpy.ndarray`, bias weights
        """
        return self.loaded[self.model_name]["elmo_softmax_b"]

    @staticmethod
    def load_vocab(
        vocab_path: str, cutoff_vocab: Optional[int] = None, verbose: bool = False
    ) -> Tuple[List[str], List[int]]:
        """
        Load vocabulary. Remove stop words from it.

        Args:
            vocab_path: path to the vocabulary file
            cutoff_vocab: maximum number of words to use, if None walk through th whole vocabulary.
            verbose: whether to print misc information

        Returns:
            ELMo vocabulary and word indexes that will be removed from LM head matrix (softmax weights).
        """
        stop_words = {"<UNK>", "<S>", "</S>", "--", "..", "...", "...."}
        word_forms_vocab = list()
        remove_ids = set()
        if verbose:
            logger.info("Reading ELMo vocabulary")

        # Read original and lemmatized vocabularies and collect Stop Word ids
        with open(vocab_path, encoding="utf-8") as f1:
            for idx, word_form in enumerate(f1):
                if cutoff_vocab is not None and idx == cutoff_vocab:
                    break
                word_form = word_form.strip()
                if len(word_form) == 1 or word_form in stop_words:
                    remove_ids.add(idx)
                else:
                    word_forms_vocab.append(word_form)

        if verbose:
            logger.info("Reading ELMo vocabulary done!")
        return word_forms_vocab, list(remove_ids)

    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        If `self.embedding_similarity` is true will return similarity scores.
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
            log_probs = self.get_emb_similarity(tokens_lists, target_ids)
        else:
            log_probs = self.predict(tokens_lists, target_ids)
        return log_probs, self.word2id

    def compute_elmo_pre_softmax(
            self,
            tokens_lists: List[List[str]],
            target_ids: List[int],
            lm_direction: str
    ) -> np.ndarray:
        """
        Runs the ELMo language model and computes rnn outputs from forward or backward pass.
        The obtained outputs are then multiplied by a softmax matrix.
        Args:
            tokens_lists: list of tokenized sentences
            target_ids: target word indexes
            lm_direction: outputs from "forward" or "backward" language model
        Returns: Logits from "forward" or "backward" pass
        """
        contexts = []
        for tokens, target_idx in zip(tokens_lists, target_ids):
            if lm_direction == "forward":
                sentence = tokens[:target_idx]
            elif lm_direction == "backward":
                sentence = tokens[target_idx + 1:]
            else:
                raise ValueError("Incorrect 'context' value: it must be 'left' or 'right'")
            if not sentence:
                sentence = ["."]
            contexts.append(sentence)

        bidirectional_rnn_outputs = list(self.model.embed_sentences(contexts, self.batch_size))

        rnn_outputs = []
        for idx in range(len(target_ids)):
            if lm_direction == "forward":
                # Output from last token
                rnn_outputs.append(
                    bidirectional_rnn_outputs[idx][ELMO_LSTM_LAYER_NUMS[-1], -1, :ELMO_HIDDEN_SIZE]
                )
            else:
                # Output from first token
                rnn_outputs.append(
                    bidirectional_rnn_outputs[idx][ELMO_LSTM_LAYER_NUMS[-1], 0, ELMO_HIDDEN_SIZE:]
                )

        return self.compute_logits(np.vstack(rnn_outputs))

    def predict(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> np.ndarray:
        """
        Get log probability distribution over vocabulary.

        Args:
            tokens_lists: list of tokenized sentences
            target_ids: target word indexes

        Returns:
            `numpy.ndarray`, matrix with rows - log-prob distribution over vocabulary.
        """
        if self.direction == "forward":
            return self.compute_elmo_pre_softmax(
                tokens_lists, target_ids, lm_direction="forward"
            )
        elif self.direction == "backward":
            return self.compute_elmo_pre_softmax(
                tokens_lists, target_ids, lm_direction="backward"
            )
        elif self.direction == "both":
            fwd_logits = self.compute_elmo_pre_softmax(
                tokens_lists, target_ids, lm_direction="forward"
            )
            bwd_logits = self.compute_elmo_pre_softmax(
                tokens_lists, target_ids, lm_direction="backward"
            )
            log_probs = fast_np_sparse_batch_combine_two_dists(fwd_logits, bwd_logits)
            return log_probs
        raise ValueError(f"Unknown variant of elmo usage: {self.direction}.")

    def compute_logits(self, states: np.ndarray) -> np.ndarray:
        """
        Compute logits of given states

        Args:
            states: numpy array with shape (num_samples, ELMO_HIDDEN_SIZE)
                Each row of this matrix corresponds to state of target word from original sentence

        Returns: logits: numpy array with shape (num_samples, vocab_size)
            This matrix is result of multiplication @states on @self.elmo_softmax_w
            @self.elmo_softmax_w: numpy array with shape (ELMO_HIDDEN_SIZE, vocab_size)
            @self.elmo_softmax_b: numpy array with shape (1, vocab_size)
        """
        elmo_softmax_w = self.loaded[self.loaded_name]["elmo_softmax_w"]
        elmo_softmax_b = self.loaded[self.loaded_name]["elmo_softmax_b"]
        logits = np.matmul(states, elmo_softmax_w)
        if self.add_bias:
            logits += elmo_softmax_b
        return logits

    def get_model_part_paths(self, model_name_or_path: str) -> Tuple[Path, ...]:
        """
        Get path to the model parts by name or exact path to model directory.

        Args:
            model_name_or_path: model or name or exact path to the model folder.

        Returns:
            path to model options, weights, softmax weights and vocab files.
        """
        if model_name_or_path in ELMO_PRETRAINED_MODEL_ARCHIVE_MAP:
            model_path = load_elmo_model(model_name_or_path, weights_dir=self.weights_path)
        else:
            model_path = Path(model_name_or_path)
        return (
            model_path / "options.json",
            model_path / "weights.hdf5",
            model_path / "softmax_weights.hdf5",
            model_path / "vocab.txt",
        )


def copy_or_download(src: str, dst: str, copy: bool = True):
    """
    Loads src file to dst directory by wget from src url or cp from src posix path
    Args:
        src: url or posix path
        dst: destination file path
        copy: indicates how to get the source file, by wget or cp command
    """

    if copy:
        shutil.copy(src, dst)
    else:
        wget.download(src, dst)


def load_elmo_model(
    model_name: str,
    cache_dir: Union[str, Path] = CACHE_DIR,
    weights_dir: str = None,
) -> Path:
    """
    Loads ELMo model if needed.

    Args:
        model_name: name of the ELMo model to be loaded.
        cache_dir: path to the cache directory where model will be stored.

    Returns:
        path to the model directory
    """
    if model_name not in ELMO_PRETRAINED_MODEL_ARCHIVE_MAP:
        raise ValueError(
            f"Wrong model name: {model_name}, "
            f"choose one from {ELMO_PRETRAINED_MODEL_ARCHIVE_MAP.keys()}."
        )
    model_cache_path = cache_dir / "resources" / model_name
    if not model_cache_path.exists():
        model_cache_path.mkdir(parents=True, exist_ok=False)

    model_urls = ELMO_PRETRAINED_MODEL_ARCHIVE_MAP[model_name]
    copy_flag = model_urls["copy"]
    options_path = model_urls["options_path"]
    weights_path = model_urls["weights_path"]
    softmax_weights_path = model_urls["softmax_weights_path"]
    vocab_path = model_urls["vocab_path"]
    if copy_flag:
        if weights_dir is None:
            weights_dir = RESOURCES_DIR
        weights_dir = Path(weights_dir)
        options_path = weights_dir / options_path
        weights_path = weights_dir / weights_path
        softmax_weights_path = weights_dir / softmax_weights_path
        vocab_path = weights_dir / vocab_path

    if not (model_cache_path / 'options.json').exists():
        logger.info("Downloading options file...")
        copy_or_download(
            options_path,
            str(model_cache_path / 'options.json'),
            copy_flag,
        )
    if not (model_cache_path / 'weights.hdf5').exists():
        logger.info("\nDownloading weights file...")
        copy_or_download(
            weights_path,
            str(model_cache_path / 'weights.hdf5'),
            copy_flag,
        )
    if not (model_cache_path / 'softmax_weights.hdf5').exists():
        logger.info("\nDownloading softmax weights file...")
        copy_or_download(
            softmax_weights_path,
            str(model_cache_path / 'softmax_weights.hdf5'),
            copy_flag,
        )
    if not (model_cache_path / 'vocab.txt').exists():
        logger.info("\nDownloading vocabulary file...")
        copy_or_download(
            vocab_path,
            str(model_cache_path / 'vocab.txt'),
            copy_flag,
        )
    return model_cache_path
