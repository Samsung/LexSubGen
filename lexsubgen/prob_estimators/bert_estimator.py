import json
import os
from string import punctuation
from typing import NoReturn, Dict, List, Tuple

import numpy as np
import torch
from overrides import overrides
from transformers import BertTokenizer, BertForMaskedLM

from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator


class BertProbEstimator(EmbSimProbEstimator):
    def __init__(
        self,
        mask_type: str = "not_masked",
        model_name: str = "bert-base-cased",
        embedding_similarity: bool = False,
        temperature: float = 1.0,
        use_attention_mask: bool = True,
        cuda_device: int = -1,
        sim_func: str = "dot-product",
        use_subword_mean: bool = False,
        verbose: bool = False,
    ):
        """
        Probability estimator based on the BERT model.
        See J. Devlin et al. "BERT: Pre-training of Deep
        Bidirectional Transformers for Language Understanding".

        Args:
            mask_type: the target word masking strategy.
            model_name: BERT model name, see https://github.com/huggingface/transformers
            embedding_similarity: whether to compute BERT embedding similarity instead of the full model
            temperature: temperature by which to divide log-probs
            use_attention_mask: whether to zero out attention on padding tokens
            cuda_device: CUDA device to load model to
            sim_func: name of similarity function to use in order to compute embedding similarity
            use_subword_mean: how to handle words that are splitted into multiple subwords when computing
            embedding similarity
            verbose: whether to print misc information
        """
        super(BertProbEstimator, self).__init__(
            model_name=model_name,
            temperature=temperature,
            sim_func=sim_func,
            verbose=verbose,
        )
        self.mask_type = mask_type
        self.embedding_similarity = embedding_similarity
        self.use_attention_mask = use_attention_mask
        self.use_subword_mean = use_subword_mean

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        if cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.descriptor = {
            "Prob_estimator": {
                "name": "bert",
                "class": self.__class__.__name__,
                "model_name": self.model_name,
                "mask_type": self.mask_type,
                "embedding_similarity": self.embedding_similarity,
                "temperature": self.temperature,
                "use_attention_mask": self.use_attention_mask,
                "use_subword_mean": self.use_subword_mean,
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
            `transformers.BertTokenizer` tokenzier related to the model
        """
        return self.loaded[self.model_name]["tokenizer"]

    def register_model(self) -> NoReturn:
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        if self.model_name not in BertProbEstimator.loaded:
            bert_model = BertForMaskedLM.from_pretrained(self.model_name)
            bert_model.to(self.device).eval()
            bert_tokenizer = BertTokenizer.from_pretrained(
                self.model_name, do_lower_case=self.model_name.endswith("uncased")
            )
            bert_word2id = BertProbEstimator.load_word2id(bert_tokenizer)
            bert_filter_word_ids = BertProbEstimator.load_filter_word_ids(
                bert_word2id, punctuation
            )
            word_embeddings = (
                bert_model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
            )
            BertProbEstimator.loaded[self.model_name] = {
                "model": bert_model,
                "tokenizer": bert_tokenizer,
                "embeddings": word_embeddings,
                "word2id": bert_word2id,
                "filter_word_ids": bert_filter_word_ids,
            }
            BertProbEstimator.loaded[self.model_name]["ref_count"] = 1
        else:
            BertProbEstimator.loaded[self.model_name]["ref_count"] += 1

    @overrides
    def get_unk_word_vector(self, word) -> np.ndarray:
        """
        This method returns vector to be used as a default if
        word is not present in the vocabulary. If `self.use_subword_mean` is true
        then the word will be splitted into subwords and mean of their embeddings
        will be taken.

        Args:
            word: word for which the vector should be given

        Returns:
            zeros vector
        """
        if self.use_subword_mean:
            sub_token_ids = self.tokenizer.encode(word)[1:-1]
            mean_vector = self.embeddings[sub_token_ids, :].mean(axis=0, keepdims=True)
            return mean_vector
        return super(BertProbEstimator, self).get_unk_word_vector(word)

    @staticmethod
    def load_word2id(tokenizer: BertTokenizer) -> Dict[str, int]:
        """
        Loads model vocabulary in the form of mapping from words to their indexes.

        Args:
            tokenizer: `transformers.BertTokenizer` tokenizer

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

    def bert_tokenize_sentence(
        self, tokens: List[str], tokenizer: BertTokenizer = None
    ) -> List[str]:
        """
        Auxiliary function that tokenize given context into subwords.

        Args:
            tokens: list of unsplitted tokens.
            tokenizer: tokenizer to be used for words tokenization into subwords.

        Returns:
            list of newly acquired tokens
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        bert_tokens = list()
        for token in tokens:
            bert_tokens.extend(tokenizer.tokenize(token))
        return bert_tokens

    def bert_prepare_batch(
        self,
        batch_of_tokens: List[List[str]],
        batch_of_target_ids: List[int],
        tokenizer: BertTokenizer = None,
    ) -> Tuple[List[List[str]], List[int]]:
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

        bert_batch_of_tokens, bert_batch_of_target_ids = list(), list()
        max_seq_len = 0
        for tokens, target_idx in zip(batch_of_tokens, batch_of_target_ids):
            left_context = ["[CLS]"] + self.bert_tokenize_sentence(
                tokens[:target_idx], tokenizer
            )
            right_context = self.bert_tokenize_sentence(
                tokens[target_idx + 1 :], tokenizer
            ) + ["[SEP]"]
            target_tokens = self.bert_tokenize_sentence([tokens[target_idx]], tokenizer)

            if self.mask_type == "masked":
                target_tokens = ["[MASK]"]
            elif self.mask_type == "combined" and len(target_tokens) > 1:
                target_tokens = ["[MASK]"]
            elif self.mask_type == "first_subtoken":
                target_tokens = target_tokens[:1]
            elif self.mask_type != "not_masked":
                raise ValueError(f"Unrecognised masking type {self.mask_type}.")

            context = left_context + target_tokens + right_context
            seq_len = len(context)
            if seq_len > max_seq_len:
                max_seq_len = seq_len

            bert_batch_of_tokens.append(context)
            bert_batch_of_target_ids.append(len(left_context))
        bert_batch_of_tokens = [
            tokens + ["[PAD]"] * (max_seq_len - len(tokens))
            for tokens in bert_batch_of_tokens
        ]
        return bert_batch_of_tokens, bert_batch_of_target_ids

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
        bert_tokens, bert_target_ids = self.bert_prepare_batch(tokens_lists, target_ids)

        input_ids = np.vstack(
            [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in bert_tokens]
        )
        input_ids = torch.tensor(input_ids).to(self.device)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).type(
                torch.FloatTensor
            )
            attention_mask = attention_mask.to(input_ids)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            logits = np.vstack(
                [
                    logits[idx, target_idx, :].cpu().numpy() / self.temperature
                    for idx, target_idx in enumerate(bert_target_ids)
                ]
            )
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
        logits[:, self.filter_word_ids] = -1e9
        return logits, self.word2id
