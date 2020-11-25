import json
import os
from string import punctuation
from typing import NoReturn, Dict, List, Tuple

import numpy as np
import torch
from overrides import overrides
from transformers import RobertaTokenizer, RobertaForMaskedLM

from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator


class RobertaProbEstimator(EmbSimProbEstimator):
    def __init__(
        self,
        mask_type: str = "not_masked",
        model_name: str = "roberta-large",
        embedding_similarity: bool = False,
        temperature: float = 1.0,
        use_attention_mask: bool = True,
        cuda_device: int = -1,
        sim_func: str = "dot-product",
        unk_word_embedding: str = "first_subtoken",
        filter_vocabulary_mode: str = "none",
        verbose: bool = False,
    ):
        """
        Probability estimator based on the Roberta model.
        See Y. Liu et al. "RoBERTa: A Robustly Optimized
        BERT Pretraining Approach".

        Args:
            mask_type: the target word masking strategy.
            model_name: Roberta model name, see https://github.com/huggingface/transformers
            embedding_similarity: whether to compute BERT embedding similarity instead of the full model
            temperature: temperature by which to divide log-probs
            use_attention_mask: whether to zero out attention on padding tokens
            cuda_device: CUDA device to load model to
            sim_func: name of similarity function to use in order to compute embedding similarity
            unk_word_embedding: how to handle words that are splitted into multiple subwords when computing
            embedding similarity
            verbose: whether to print misc information
        """
        super(RobertaProbEstimator, self).__init__(
            model_name=model_name,
            temperature=temperature,
            sim_func=sim_func,
            verbose=verbose,
        )
        self.mask_type = mask_type
        self.embedding_similarity = embedding_similarity
        self.use_attention_mask = use_attention_mask
        self.unk_word_embedding = unk_word_embedding
        self.filter_vocabulary_mode = filter_vocabulary_mode
        self.prev_word2id = {}

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        if cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.descriptor = {
            "Prob_estimator": {
                "name": "roberta",
                "class": self.__class__.__name__,
                "model_name": self.model_name,
                "mask_type": self.mask_type,
                "embedding_similarity": self.embedding_similarity,
                "temperature": self.temperature,
                "use_attention_mask": self.use_attention_mask,
                "unk_word_embedding": self.unk_word_embedding,
            }
        }

        self.register_model()

        self.logger.debug(f"Probability estimator {self.descriptor} is created.")
        self.logger.debug(f"Config:\n{json.dumps(self.descriptor, indent=4)}")

    @property
    def tokenizer(self):
        """
        Model tokenizer.

        Returns:
            `transformers.RobertaTokenizer` tokenzier related to the model
        """
        return self.loaded[self.model_name]["tokenizer"]

    @property
    def parameters(self):
        parameters = f"{self.mask_type}{self.model_name}" \
                     f"{self.use_attention_mask}{self.filter_vocabulary_mode}"

        if self.embedding_similarity:
            parameters += f"embs{self.unk_word_embedding}{self.sim_func}"

        return parameters

    def register_model(self) -> NoReturn:
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        if self.model_name not in RobertaProbEstimator.loaded:
            roberta_model = RobertaForMaskedLM.from_pretrained(self.model_name)
            roberta_model.to(self.device).eval()
            roberta_tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            roberta_word2id = RobertaProbEstimator.load_word2id(roberta_tokenizer)
            filter_word_ids = RobertaProbEstimator.load_filter_word_ids(
                roberta_word2id, punctuation
            )
            word_embeddings = (
                roberta_model.lm_head.decoder.weight.data.cpu().numpy()
            )

            norms = np.linalg.norm(word_embeddings, axis=-1, keepdims=True)
            normed_word_embeddings = word_embeddings / norms

            RobertaProbEstimator.loaded[self.model_name] = {
                "model": roberta_model,
                "tokenizer": roberta_tokenizer,
                "embeddings": word_embeddings,
                "normed_embeddings": normed_word_embeddings,
                "word2id": roberta_word2id,
                "filter_word_ids": filter_word_ids,
            }
            RobertaProbEstimator.loaded[self.model_name]["ref_count"] = 1
        else:
            RobertaProbEstimator.loaded[self.model_name]["ref_count"] += 1

    @property
    def normed_embeddings(self) -> np.ndarray:
        """
        Attribute that acquires model word normed_embeddings.

        Returns:
            2-D `numpy.ndarray` with rows representing word vectors.
        """
        return self.loaded[self.model_name]["normed_embeddings"]

    def get_emb_similarity(
        self, tokens_batch: List[List[str]], target_ids_batch: List[int],
    ) -> np.ndarray:
        """
        Computes similarity between each target word and substitutes
        according to their embedding vectors.

        Args:
            tokens_batch: list of contexts
            target_ids_batch: list of target word ids in the given contexts

        Returns:
            similarity scores between target words and
            words from the model vocabulary.
        """
        if self.sim_func == "dot-product":
            embeddings = self.embeddings
        else:
            embeddings = self.normed_embeddings

        target_word_embeddings = []
        for tokens, pos in zip(tokens_batch, target_ids_batch):
            tokenized = self.tokenize_around_target(tokens, pos, self.tokenizer)
            _, _, target_subtokens_ids = tokenized

            target_word_embeddings.append(
                self.get_target_embedding(target_subtokens_ids, embeddings)
            )

        target_word_embeddings = np.vstack(target_word_embeddings)
        emb_sim = np.matmul(target_word_embeddings, embeddings.T)

        return emb_sim / self.temperature

    def get_target_embedding(
        self,
        target_subtokens_ids: List[int],
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Returns an embedding that will be used if the given word is not presented
        in the vocabulary. The word is split into subwords and depending on the
        self.unk_word_embedding parameter the final embedding is built.

        Args:
            word: word for which the vector should be given
            target_subtokens_ids: vocabulary indexes of target word subtokens
            embeddings: roberta embeddings of target word subtokens

        Returns:
            embedding of the unknown word
        """
        if self.unk_word_embedding == "mean":
            return embeddings[target_subtokens_ids].mean(axis=0, keepdims=True)
        elif self.unk_word_embedding == "first_subtoken":
            return embeddings[target_subtokens_ids[0]]
        elif self.unk_word_embedding == "last_subtoken":
            return embeddings[target_subtokens_ids[-1]]
        else:
            raise ValueError(
                f"Incorrect value of unk_word_embedding: "
                f"{self.unk_word_embedding}"
            )

    @staticmethod
    def load_word2id(tokenizer: RobertaTokenizer) -> Dict[str, int]:
        """
        Loads model vocabulary in the form of mapping from words to their indexes.

        Args:
            tokenizer: `transformers.RobertaTokenizer` tokenizer

        Returns:
            model vocabulary
        """
        word2id = dict()
        for word_idx in range(tokenizer.vocab_size):
            word = tokenizer.convert_ids_to_tokens([word_idx])[0]
            word2id[word] = word_idx
        return word2id

    @staticmethod
    def load_filter_word_ids(word2id: Dict[str, int], filter_chars: str) -> List[int]:
        """
        Gathers words that should be filtered from the end distribution, e.g.
        punctuation.

        Args:
            word2id: model vocabulary
            filter_chars: words with this chars should be filtered from end distribution.

        Returns:
            Indexes of words to be filtered from the end distribution.
        """
        filter_word_ids = []
        set_filter_chars = set(filter_chars)
        for word, idx in word2id.items():
            if len(set(word) & set_filter_chars):
                filter_word_ids.append(idx)
        return filter_word_ids

    @property
    def filter_word_ids(self) -> List[int]:
        """
        Indexes of words to be filtered from the end distribution.

        Returns:
            list of indexes
        """
        return self.loaded[self.model_name]["filter_word_ids"]

    @staticmethod
    def tokenize_around_target(
        tokens: List[str],
        target_idx: int,
        tokenizer: RobertaTokenizer = None,
    ):
        left_specsym_len = 1  # for BERT / ROBERTA there is 1 spec token before text
        input_text = ' '.join(tokens)
        tokenized_text = tokenizer.encode(' ' + input_text, add_special_tokens=True)

        left_ctx = ' '.join(tokens[:target_idx])
        target_start = left_specsym_len + len(tokenizer.encode(
            ' ' + left_ctx, add_special_tokens=False
        ))

        left_ctx_target = ' '.join(tokens[:target_idx + 1])
        target_subtokens_ids = tokenizer.encode(
            ' ' + left_ctx_target, add_special_tokens=False
        )[target_start - left_specsym_len:]

        return tokenized_text, target_start, target_subtokens_ids

    def prepare_batch(
        self,
        batch_of_tokens: List[List[str]],
        batch_of_target_ids: List[int],
        tokenizer: RobertaTokenizer = None,
    ):
        """
        Prepares batch of contexts and target indexes into the form
        suitable for processing with BERT, e.g. tokenziation, addition of special tokens
        like [CLS] and [SEP], padding contexts to have the same size etc.

        Args:
            batch_of_tokens: list of contexts
            batch_of_target_ids: list of target word indexes
            tokenizer: tokenizer to use for word tokenization

        Returns:
            transformed contexts and target word indexes in these new contexts
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        roberta_batch_of_tokens, roberta_batch_of_target_ids = [], []
        max_seq_len = 0
        for tokens, target_idx in zip(batch_of_tokens, batch_of_target_ids):
            tokenized = self.tokenize_around_target(tokens, target_idx, tokenizer)
            context, target_start, target_subtokens_ids = tokenized

            if self.mask_type == "masked":
                context = context[:target_start] + \
                          [tokenizer.mask_token_id] + \
                          context[target_start + len(target_subtokens_ids):]
            elif self.mask_type != "not_masked":
                raise ValueError(f"Unrecognised masking type {self.mask_type}.")

            if len(context) > 512:
                first_subtok = context[target_start]
                # Cropping maximum context around the target word
                left_idx = max(0, target_start - 256)
                right_idx = min(target_start + 256, len(context))
                context = context[left_idx: right_idx]
                target_start = target_start if target_start < 256 else 255
                assert first_subtok == context[target_start]

            max_seq_len = max(max_seq_len, len(context))

            roberta_batch_of_tokens.append(context)
            roberta_batch_of_target_ids.append(target_start)

        assert max_seq_len <= 512

        input_ids = np.vstack([
            tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
            for tokens in roberta_batch_of_tokens
        ])

        input_ids = torch.tensor(input_ids).to(self.device)

        return input_ids, roberta_batch_of_target_ids

    def predict(
        self, tokens_lists: List[List[str]], target_ids: List[int],
    ) -> np.ndarray:
        """
        Get log probability distribution over vocabulary.

        Args:
            tokens_lists: list of contexts
            target_ids: target word indexes

        Returns:
            `numpy.ndarray`, matrix with rows - log-prob distribution over vocabulary.
        """
        input_ids, mod_target_ids = self.prepare_batch(tokens_lists, target_ids)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = (input_ids != self.tokenizer.pad_token_id)
            attention_mask = attention_mask.float().to(input_ids)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            logits = np.vstack([
                logits[idx, target_idx, :].cpu().numpy()
                for idx, target_idx in enumerate(mod_target_ids)
            ])
            return logits

    @overrides
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
            logits = self.get_emb_similarity(tokens_lists, target_ids)
        else:
            logits = self.predict(tokens_lists, target_ids)

        return logits, self.word2id
