import os
import gc
from typing import NoReturn, Dict, List, Tuple

import string
import re

import numpy as np
import torch
from overrides import overrides

from lexsubgen.prob_estimators.base_estimator import BaseProbEstimator

from transformers import T5Tokenizer, MT5ForConditionalGeneration

from word_forms.word_forms import get_word_forms

DEBUG_FILE = 'debug/mt5.txt'


def get_span(string):
    """
        Extract span from single output model decoded.
        Example: get_span('<extra_id_0> very professional <extra_id_1>')->'very professional'
    """
    string = string.replace('<extra_id_0>', '')
    string = string.replace('<extra_id_1>', '')
    return string.strip()


def del_punc(string_in):
    """
        Delete punctuation from string. Punctuation from string.punctuation
    """
    # Delete lower for Germany!!!
    return re.sub('([{}])'.format(string.punctuation), r'', string_in).strip().lower()


def max_length(span):
    """
        Strategy "max_length" for extract single word from span.
        The word maximum in length is selected.
    """
    array = span.split()
    if array:
        return del_punc(max(array, key=lambda x: len(x)))
    else:
        return del_punc(span)


def last_word(span):
    """
        Strategy "last_word" for extract single word from span.
        The last word in span is selected.
    """
    if span:
        return del_punc(span.split()[-1])
    else:
        return del_punc(span)


def print_subsequences(subseq, length):
    """
        Function for debug. It's print a lot of information about generating sequences
    """
    with open(DEBUG_FILE, 'a') as file_debug:
        if length == 1:
            print('>' + ';'.join([f"'{el[0][0]}' {np.exp(el[0][1]): .3f} " for el in subseq]), file=file_debug,
                  flush=True)
        else:
            for_print = []
            ind_pad = length

            for i in range(len(subseq)):
                # '<pad>'
                subwords, probs = zip(*subseq[i])
                if (length > 2) and (subwords[-2] == '<pad>'):
                    continue
                if '<pad>' in subwords:
                    ind_pad = subwords.index('<pad>')
                preds = ' '.join(subwords[:-1])
                print(f'{preds} -> {subwords[-1]} [{np.exp(probs[-1]):.5f}]', file=file_debug, flush=True)
                for_print.append("'" + ' '.join(subwords) + "'" + f' {np.exp(np.sum(probs[:ind_pad]) / ind_pad): .3f}')
            print('>' + ';'.join(for_print), file=file_debug, flush=True)


class MT5ProbEstimator(BaseProbEstimator):
    def __init__(
            self,
            model_name="google/mt5-small",
            cuda_device: int = -1,
            num_beams: int = 1,
            num_return_sequences: int = 1,
            handler_span: str = None,
            exclude_target: bool = False,
            debug: bool = False,
            is_english_forms: bool = False,
            eos_token: str = '<extra_id_1>',
            verbose: bool = False
    ):
        """
        Probability estimator based on the mT5 model.
        See Linting Xue et al. "mT5: A massively
        multilingual pre-trained text-to-text transformer"

        Args:
            model_name:
                Version of model
            cuda_device:
                Whether compute on gpu
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            handler_span:
                Strategy processing substitution, including few words
            exclude_target:
                Whether to exclude target word from generating sequences
            debug:
                Whether to print detail information about generating sequences
            is_english_forms:
                If is_english_forms=True then prohibit to generate all english targets word forms for English
            eos_token:
                The id of the *end-of-sequence* token.
            verbose:
                whether to print misc information
        """
        super().__init__(verbose=verbose)
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.handler_span = handler_span
        self.exclude_target = exclude_target
        self.debug = debug
        self.is_english_forms = is_english_forms

        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:5120'
        if cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.logger.info("Downloading model...")
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True)
        self.logger.info("Model download has finished.")
        self.model.to(self.device)
        gc.collect()

        self.logger.info("Downloading tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.logger.info("Tokenizer download has finished.")

        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(eos_token)

        if self.debug:
            self.w2id = self.tokenizer.get_vocab()
            self.id2w = {v: k for (k, v) in self.w2id.items()}

    @overrides
    def get_log_probs(
            self, tokens_lists: List[List[str]], target_ids: List[int]
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
        batch_size = len(tokens_lists)
        exclude_tokenized_targets = None

        if self.exclude_target:
            # if exclude_target=True then it's duplicate mode!
            exclude_targets = set()
            for token_list, target_id in zip(tokens_lists, target_ids):
                len_sentence = len(token_list) // 2
                exclude_targets.add(token_list[target_id - len_sentence])
            exclude_tokenized_targets = []

            if self.is_english_forms:
                for target in exclude_targets:
                    union_forms = set()
                    word_forms = get_word_forms(target)
                    for key in word_forms:
                        union_forms = union_forms.union(word_forms[key])
                    union_forms = list(union_forms)
                    if union_forms:
                        tokenized = self.tokenizer(union_forms, add_special_tokens=False).input_ids
                        for i in range(len(tokenized)):
                            while 259 in tokenized[i]:
                                tokenized[i].remove(259)
                        exclude_tokenized_targets.extend(tokenized)
            else:
                for target in exclude_targets:
                    # 259 is subword: '▁'
                    tokenized = self.tokenizer(target, add_special_tokens=False).input_ids
                    while 259 in tokenized:
                        tokenized.remove(259)
                    if len(tokenized) > 1:
                        tokenized = [[tokenized[i]] for i in range(len(tokenized) - 1)]
                        res_tmp = []
                        for el in tokenized:
                            res_tmp.append(tuple(sorted(el)))
                        tokenized = [list(el) for el in list(set(res_tmp))]
                    else:
                        tokenized = [tokenized]
                    exclude_tokenized_targets.extend(tokenized)

            if not exclude_tokenized_targets:
                exclude_tokenized_targets = None

        encoding = self.tokenizer(tokens_lists,
                                  is_split_into_words=True,
                                  padding=True,
                                  return_tensors="pt")

        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids,
                                          attention_mask=attention_mask,
                                          num_beams=self.num_beams,
                                          num_return_sequences=self.num_return_sequences,
                                          bad_words_ids=exclude_tokenized_targets,
                                          min_length=2,
                                          length_penalty=1.0,
                                          output_scores=True,
                                          early_stopping=True,
                                          return_dict_in_generate=True,
                                          eos_token_id=self.eos_token_id
                                          )  # num_beams >= num_return_sequences

            all_words = []
            word2id = {}
            id2word = {}
            cur_id = 0

            # outputs['sequences'].shape=(num_batch * num_return_sequences, max_generating_sequence_length)
            for i in range(outputs['sequences'].shape[0]):
                span = get_span(self.tokenizer.decode(outputs['sequences'][i], skip_special_tokens=True))
                if self.handler_span == 'max_length':
                    span = max_length(span)
                elif self.handler_span == 'last_word':
                    span = last_word(span)
                all_words.append(span)
                if span not in word2id:
                    word2id[span] = cur_id
                    id2word[cur_id] = span
                    cur_id += 1

            log_probs = np.zeros((batch_size, len(word2id))) - 1e200
            id_cur_element = -1

            # outputs['sequences_scores'].shape=num_batch * num_return_sequences
            for i, span in enumerate(all_words):
                if i % self.num_return_sequences == 0:
                    id_cur_element += 1
                log_probs[id_cur_element][word2id[span]] = outputs['sequences_scores'][i]

            if self.debug:
                with open(DEBUG_FILE, 'a') as file_debug:
                    for i in range(batch_size):
                        print('-->', ' '.join(tokens_lists[i]), file=file_debug, flush=True)
                        tmp = [self.id2w[id.item()] for id in input_ids[i]]
                        print('/'.join(tmp), file=file_debug, flush=True)
                        sequences = [[] for _ in range(self.num_return_sequences)]
                        for j in range(1, outputs['sequences'].shape[1]):
                            print(f'Filling mask {j}', file=file_debug, flush=True)
                            for k in range(self.num_return_sequences):
                                ind_subword = outputs['sequences'][i * batch_size + k][j].item()
                                beam_ind = outputs['beam_indices'][i * batch_size + k][j - 1].item()
                                score_subword = outputs['scores'][j - 1][beam_ind][ind_subword].item()
                                sequences[k].append((self.id2w[ind_subword], score_subword))
                            print_subsequences(sequences, j)

            return log_probs, word2id
