import logging
import os
import random
from string import punctuation
from pathlib import Path
from typing import List, Tuple, Dict, NoReturn

import numpy as np
import torch
from overrides import overrides
from transformers import XLNetLMHeadModel, XLNetTokenizer, SPIECE_UNDERLINE

from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

logger = logging.getLogger(Path(__file__).name)


class XLNetProbEstimator(EmbSimProbEstimator):
    def __init__(
        self,
        model_name: str = "xlnet-large-cased",
        masked: bool = True,
        use_input_mask: bool = False,
        embedding_similarity: bool = False,
        temperature: float = 1.0,
        cuda_device: int = -1,
        multi_subword: bool = False,
        top_k_subword: int = 10,
        filter_words: bool = True,
        sim_func: str = "dot-product",
        use_subword_mean: bool = False,
        verbose: bool = False,
    ):
        """
        Probability estimator based on XLNet model, see
        Z. Yang et al. "XLNet: Generalized Autoregressive Pretraining
        for Language Understanding".

        Args:
            model_name: name of the XLNet model, see https://github.com/huggingface/transformers
            masked: whether to mask target word or not
            use_input_mask: whether to zero out attention weights for pad tokens
            embedding_similarity: whether to compute XLNet embedding similarity instead of the full model
            temperature: temperature by which to divide log-probs
            cuda_device: CUDA device to load model to
            multi_subword: whether to generate multi-subword words
            top_k_subword: branching factor when generating multi-subword words
            filter_words: whether to filter special tokens and word pieces
            sim_func: name of similarity function to use in order to compute embedding similarity
            use_subword_mean: how to handle words that are splitted into multiple subwords when computing
            verbose: whether to print misc information
        """
        super(XLNetProbEstimator, self).__init__(
            model_name=model_name,
            verbose=verbose,
            sim_func=sim_func,
            temperature=temperature,
        )
        self.cuda_device = cuda_device
        self.use_input_mask = use_input_mask
        self.masked = masked
        self.multi_subword = multi_subword
        self.top_k_subword = top_k_subword
        self.filter_words = filter_words
        self.embedding_similarity = embedding_similarity
        self.use_subword_mean = use_subword_mean

        self.descr = {
            "Prob_generator": {
                "name": "xlnet",
                "model_name": self.model_name,
                "use_input_mask": self.use_input_mask,
                "masked": self.masked,
                "multi_subword": self.multi_subword,
                "use_subword_mean": self.use_subword_mean,
            }
        }

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        if cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.NON_START_SYMBOL = "##"
        self.register_model()

    def register_model(self) -> NoReturn:
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        if self.model_name not in XLNetProbEstimator.loaded:
            model = XLNetLMHeadModel.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            tokenizer = XLNetTokenizer.from_pretrained(self.model_name)
            word2id = self._get_word2id(tokenizer)
            spiece_ids = [
                idx
                for word, idx in word2id.items()
                if word.startswith(self.NON_START_SYMBOL)
            ]
            all_special_ids = tokenizer.all_special_ids
            word_embeddings = model.transformer.word_embedding.weight.data.cpu().numpy()
            XLNetProbEstimator.loaded[self.model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "embeddings": word_embeddings,
                "word2id": word2id,
                "spiece_ids": spiece_ids,
                "all_special_ids": all_special_ids,
            }
            XLNetProbEstimator.loaded[self.model_name]["ref_count"] = 1
        else:
            XLNetProbEstimator.loaded[self.model_name]["ref_count"] += 1

    @property
    def spiece_ids(self):
        """
        Indexes of word pieces, i.e. words that start with special token
        (in original tokenizer that words doesn't start with special underline
        score token so they are non-starting parts of some words). We filter them
        cause they do not represent any word from a target vocabulary.

        Returns:
            list of indexes of word pieces.
        """
        return self.loaded[self.model_name]["spiece_ids"]

    @property
    def all_special_ids(self):
        return self.loaded[self.model_name]["all_special_ids"]

    @property
    def tokenizer(self):
        """
        Tokenizer related to the current model.

        Returns:
            `transformers.XLNetTokenizer`
        """
        return self.loaded[self.model_name]["tokenizer"]

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
            # TODO: should we exclude special symbols?
            # Getting scores according to word embeddings similarity
            logits = self.get_emb_similarity(tokens_lists, target_ids)
            return logits, self.word2id
        # Use full model to predict masked word
        logits, word2id = self.predict(tokens_lists, target_ids)
        return logits, word2id

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
            sub_token_ids = self.tokenizer.encode(word)[:-2]
            mean_vector = self.embeddings[sub_token_ids, :].mean(axis=0, keepdims=True)
            return mean_vector
        return super(XLNetProbEstimator, self).get_unk_word_vector(word)

    def predict(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Get log probability distribution over vocabulary.

        Args:
            tokens_lists: list of contexts
            target_ids: target word indexes

        Returns:
            `numpy.ndarray`, matrix with rows - log-prob distribution over vocabulary.
        """
        numerical_sentences, target_positions = self._numericalize_batch(
            tokens_lists=tokens_lists, target_ids=target_ids
        )
        input_ids, perm_mask, target_mapping, input_mask = self._prepare_inputs(
            numerical_sentences, target_positions, multi_subword=False
        )
        predictions = self.get_predictions(
            input_ids, perm_mask, target_mapping, input_mask
        )
        predictions = predictions.cpu()
        word2id = self.word2id
        if self.multi_subword:
            # TODO: check implementation of multi sub-word generation process
            predictions, word2id = self.get_multi_subword_predictions(
                predictions, numerical_sentences, target_positions
            )
        predictions = predictions.numpy()
        return predictions, word2id

    def _numericalize_batch(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Tokenize contexts and numericalize them according to model vocabulary.
        Update target token indexes in new obtained contexts.

        Args:
            tokens_lists: list of contexts
            target_ids: list of target word indexes

        Returns:
            numerical contexts and updated target word positions
        """
        numerical_sentences, target_positions = [], []
        for tokens, target_id in zip(tokens_lists, target_ids):
            seq, pos = self.get_new_token_seq_and_pos(tokens, target_id)
            numerical_sentences.append(seq)
            target_positions.append(pos)
        return numerical_sentences, target_positions

    def _get_word2id(self, tokenizer: XLNetTokenizer, convert: bool = True):
        """
        Get model vocabulary in the form of mapping from words to indexes.

        Args:
            tokenizer: model tokenizer
            convert: whether to convert words with special underline scores characters
            into ordinary words and prepend word pieces with special characters.

        Returns:
            model vocabulary
        """
        word2id = dict()
        for idx in range(tokenizer.vocab_size):
            token: str = tokenizer.convert_ids_to_tokens(idx)
            if convert:
                # Prepare vocab suitable for substitution evaluation
                # Remove sentence piece underline and add special symbol to intra word parts
                if token.startswith(SPIECE_UNDERLINE) and len(token) > 1:
                    token = token[1:]
                else:
                    token = self.NON_START_SYMBOL + token
                word2id[token] = idx
        return word2id

    def get_new_token_seq_and_pos(self, tokens: List[str], target_id: int):
        """
        Transform original context into the form suitable for processing with XLNet model.

        Args:
            tokens: context
            target_id: target word id

        Returns:
            transformed context and new target word position index
        """
        target_word = tokens[target_id]
        sentence = " ".join(
            [
                token if idx != target_id else self.tokenizer.mask_token
                for idx, token in enumerate(tokens)
            ]
        )
        sentence = self.tokenizer.clean_up_tokenization(sentence)
        sent_numerical = self.tokenizer.encode(sentence)[:-2]

        # TODO: test target position search
        def get_target_id(indexes: List[int]) -> int:
            pos = 0
            mask_id = self.tokenizer._convert_token_to_id(self.tokenizer.mask_token)

            while pos < len(indexes):
                if indexes[pos] == mask_id:
                    break
                pos += 1
            else:
                raise ValueError("Can't find masked token")
            return pos

        target_id = get_target_id(sent_numerical)
        if not self.masked:
            target_codes = self.tokenizer.encode(target_word)[:-2]
            if len(target_codes) > 1:
                if target_codes[0] == SPIECE_UNDERLINE:
                    target_codes = target_codes[1:]
            sent_numerical[target_id] = target_codes[0]
        return sent_numerical, target_id

    def get_multi_subword_predictions(
        self, predictions: torch.Tensor, sentences: List, target_ids: List
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Predict distribution with multi-subword acquistion.

        Args:
            predictions: model predictions from the last step.
            sentences: list of contexts
            target_ids: list of target word indexes

        Returns:
            predictions extended with multi-subwords and extended vocabulary
        """
        # TODO: refactor multi-subword generation process
        new_word2id = dict(self.word2id)
        extended_sentences = []
        for sentence, target_id in zip(sentences, target_ids):
            # Create new sentence with to contiguous masks
            sentence_ext = (
                sentence[:target_id]
                + [self.tokenizer.mask_token_id] * 2
                + sentence[target_id + 1 :]
            )
            extended_sentences.append(sentence_ext)
        inputs = self._prepare_inputs(
            extended_sentences, target_ids, multi_subword=True
        )
        extended_sent_predictions = self.get_predictions(*inputs, exclude_subword=True)

        new_words = []
        input_ids = inputs[0]
        for idx, target_id in enumerate(target_ids):
            (
                ext_input_ids,
                perm_mask,
                target_mapping,
                input_mask,
                top_probs,
                top_indexes,
            ) = self._create_extended_top_substitutions(
                extended_sent_predictions,
                input_ids,
                idx,
                target_id,
                top_k=self.top_k_subword,
            )
            ext_predictions = self.get_predictions(
                ext_input_ids,
                perm_mask,
                target_mapping,
                input_mask,
                exclude_subword=False,
            )
            completed_subwords = self._complete_subwords(
                top_probs, top_indexes, ext_predictions, filter_words=self.filter_words
            )
            new_words.append(completed_subwords)
            # Update word2id
            vocab_size = len(new_word2id)
            for key in completed_subwords:
                if key not in new_word2id:
                    new_word2id[key] = vocab_size
                    vocab_size += 1
        origin_vocab_size = len(self.word2id)
        vocab_size_diff = len(new_word2id) - origin_vocab_size
        if vocab_size_diff <= 0:
            return predictions, self.word2id
        # Extending predictions
        subword_predictions = torch.zeros(
            predictions.size(0), vocab_size_diff, dtype=torch.float32
        )
        for idx, words_dict in enumerate(new_words):
            for key, value in words_dict.items():
                if key not in self.word2id:
                    subword_predictions[
                        idx, new_word2id[key] - origin_vocab_size
                    ] = value
        extended_predictions = torch.cat([predictions, subword_predictions], dim=1)
        return extended_predictions, new_word2id

    def get_predictions(
        self,
        input_ids: torch.Tensor,
        perm_mask: torch.Tensor,
        target_mapping: torch.Tensor,
        input_mask: torch.Tensor,
        exclude_subword: bool = True,
    ) -> torch.Tensor:
        """
        Get XLNet model predictions for a given input.

        Args:
            input_ids: input matrix
            perm_mask: mask to indicate the attention pattern for each input token with values
                selected in ``[0, 1]``:
                If ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
                if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.
            target_mapping: mask to indicate the output tokens to use.
                If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
            input_mask: mask to avoid performing attention on padding token indices.
            exclude_subword: whether to remove subwords from final distribution (zero out probabilities)

        Returns:
            predicted distribution over vocabulary
        """
        with torch.no_grad():
            predictions = self.model(
                input_ids,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                input_mask=input_mask,
            )[0]
        predictions = predictions[:, 0, :]
        predictions = self._exclude_special_symbols(predictions)
        if exclude_subword:
            predictions = self._exclude_subword_symbols(predictions)
        return predictions

    def _complete_subwords(
        self,
        first_subword_probs: torch.Tensor,
        first_subword_indexes: torch.Tensor,
        second_subword_probs: torch.Tensor,
        filter_words: bool = True,
    ) -> Dict[str, float]:
        """
        Combine two subwords in order to get whole words. The log-probability of combination
        is the mean of their log-probs.

        Args:
            first_subword_probs: tensor containing first subwords distribution.
            first_subword_indexes: tensor containing first subword indexes.
            second_subword_probs: tensor containing second subword distribution.
            filter_words: whether to remove words with punctuation and other special tokens.

        Returns:
            mapping from predicted word to their log-probabilities
        """
        indexes_1 = first_subword_indexes.squeeze().data.cpu()
        log_probs_1 = first_subword_probs.squeeze().data.cpu()

        # TODO: which type to use for results (how to aggregate data)
        results = {}
        for i, (idx_1, log_prob_1) in enumerate(zip(indexes_1, log_probs_1)):
            log_prob_2, idx_2 = torch.topk(
                second_subword_probs[i, :].view(1, 1, -1), k=1
            )
            log_prob_2 = log_prob_2.squeeze().item()
            pred_idx_2 = idx_2.squeeze().item()
            tok_1: str = self.tokenizer.convert_ids_to_tokens(idx_1.item())
            tok_2: str = self.tokenizer.convert_ids_to_tokens(pred_idx_2)

            # filter tokens with punctuation
            if tok_2.endswith(punctuation):
                continue

            if filter_words and tok_2.startswith("▁"):
                continue

            subst = (tok_1 + tok_2).replace(SPIECE_UNDERLINE, " ").strip()
            mean_log_prob = (log_prob_1 + log_prob_2).item() / 2.0
            results[subst] = mean_log_prob
        return results

    def _create_extended_top_substitutions(
        self,
        log_probs: torch.Tensor,
        input_ids: torch.Tensor,
        idx: int,
        target_id: int,
        top_k: int = 100,
    ):
        """
        Preprocess inputs for multi-subword generation. Acquire top-k
        first subwords and their indexes. Place them in k new contexts
        and prepare inputs to predict next subword token.

        Args:
            log_probs: log-probs for first subword
            input_ids: input tensor from the previous multi-subword generation step
            idx: index of element from the batch
            target_id: index of the target word
            top_k: branching factor for multi-subword generation

        Returns:
            extended input tensor, permutation mask, target mapping, original input tensor, top k log-probs and indexes
        """
        top_log_probs, top_indexes = torch.topk(
            log_probs[idx, :].view(1, 1, -1), k=top_k
        )
        ext_input_ids = input_ids[idx].repeat((top_k, 1))
        ext_input_ids[:, target_id] = top_indexes.squeeze()
        ext_input_ids = ext_input_ids.to(self.device)
        perm_mask = torch.zeros(
            (top_k, ext_input_ids.shape[1], ext_input_ids.shape[1]),
            dtype=torch.float,
            device=self.device,
        )
        perm_mask[:, :, target_id + 1] = 1.0
        target_mapping = torch.zeros(
            (top_k, 1, ext_input_ids.shape[1]), dtype=torch.float, device=self.device
        )
        target_mapping[:, 0, target_id + 1] = 1.0
        input_mask = None
        if self.use_input_mask:
            input_mask = (ext_input_ids == self.tokenizer.pad_token_id).type(
                torch.FloatTensor
            )
            input_mask = input_mask.to(perm_mask)
        return (
            ext_input_ids,
            perm_mask,
            target_mapping,
            input_mask,
            top_log_probs,
            top_indexes,
        )

    def _prepare_inputs(
        self, tokens: List[List[int]], target_ids: List[int], multi_subword: bool
    ):
        """
        Prepare input batch for processing with XLNet model: pad contexts to have same length,
        generate permutation mask according to masking strategy, create target mapping and input mask.

        Args:
            tokens: list of contexts
            target_ids: list of target word indexes
            multi_subword: whether to generate multi-subword words

        Returns:
            input tensor, permutation mask, target mapping and input mask for `transformers.XLNetLMHead` model.
        """
        tokens_padded = self._pad_batch(tokens)
        input_ids = torch.tensor(tokens_padded)
        input_ids = input_ids.to(self.device)
        if not multi_subword:
            perm_mask, target_mapping = self._create_perm_mask_and_target_map(
                input_ids.shape[1], target_ids
            )
        else:
            perm_mask, target_mapping = self._create_perm_mask_and_target_map_sub_word(
                input_ids.shape[1], target_ids
            )
        input_mask = None
        if self.use_input_mask:
            input_mask = (input_ids == self.tokenizer.pad_token_id).type(
                torch.FloatTensor
            )
            input_mask = input_mask.to(perm_mask)
        return input_ids, perm_mask, target_mapping, input_mask

    def _pad_batch(self, token_ids: List[List[int]]) -> List[List[int]]:
        """
        Pad given batch of contexts.

        Args:
            token_ids: list of contexts

        Returns:
            list of padded contexts all having the same length
        """
        max_len = max([len(ids) for ids in token_ids])
        for ids in token_ids:
            ids.extend(
                [self.tokenizer._convert_token_to_id(self.tokenizer.pad_token)]
                * (max_len - len(ids))
            )
        return token_ids

    def _create_perm_mask_and_target_map(
        self, seq_len: int, target_ids: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Generates permutation mask and target mapping.
        If `self.masked` is true then there is no word that sees target word through attention.
        If it is false then only target word doesn't see itself.

        Args:
            seq_len: length of the sequence (context)
            target_ids: target word indexes

        Returns:
            two `torch.Tensor`s: permutation mask and target mapping
        """
        assert isinstance(target_ids[0], int), "One target per sentence"
        batch_size = len(target_ids)
        perm_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float)
        target_mapping = torch.zeros((batch_size, 1, seq_len))

        for idx, target_id in enumerate(target_ids):
            perm_mask[idx, :, target_id] = 1.0
            target_mapping[idx, 0, target_id] = 1.0
            if not self.masked:
                perm_mask[idx, :, target_id] = 0.0
                perm_mask[idx, target_id, target_id] = 1.0
        perm_mask = perm_mask.to(self.device)
        target_mapping = target_mapping.to(self.device)
        return perm_mask, target_mapping

    def _create_perm_mask_and_target_map_sub_word(
        self, seq_len: int, target_ids: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Generates permutation mask and target mapping for multi-subword geenration.
        If `self.masked` is true then there is no word that sees target word through attention.
        If it is false then only target word doesn't see itself.
        ATTENTION. Now we only support generation of words that consists of two subwords.

        Args:
            seq_len: length of the sequence (context)
            target_ids: target word indexes

        Returns:
            two `torch.Tensor`s: permutation mask and target mapping
        """
        batch_size = len(target_ids)
        perm_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float)
        target_mapping = torch.zeros((batch_size, 2, seq_len))

        for idx, target_id in enumerate(target_ids):
            perm_mask[idx, :, (target_id, target_id + 1)] = 1.0
            perm_mask[idx, target_id + 1, target_id] = 0.0
            target_mapping[idx, 0, target_id] = 1.0
            target_mapping[idx, 1, target_id + 1] = 1.0
        perm_mask = perm_mask.to(self.device)
        target_mapping = target_mapping.to(self.device)
        return perm_mask, target_mapping

    def _exclude_special_symbols(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Zero out probabilities related to special symbols e.g. punctuation.

        Args:
            predictions: original predictions

        Returns:
            filtered predictions
        """
        mask = torch.zeros(predictions.size(-1), dtype=torch.bool)
        mask[self.all_special_ids] = True
        predictions[:, mask] = -1e9
        return predictions

    def _exclude_subword_symbols(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Zero out probabilities related to subwords.

        Args:
            predictions: original predictions

        Returns:
            filtered predictions
        """
        mask = torch.zeros(predictions.size(-1), dtype=torch.bool)
        mask[self.spiece_ids] = True
        predictions[:, mask] = -1e9
        return predictions
