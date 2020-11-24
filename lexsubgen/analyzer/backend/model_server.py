import json
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
from typing import List, Callable, NoReturn

import cachetools
import fire
from _jsonnet import evaluate_file
from flask import Flask, request
from flask_cors import CORS

from lexsubgen import SubstituteGenerator
from lexsubgen.analyzer.backend.utils import (
    EndpointAction,
    SubstResponse,
    EvalResponse,
    check_and_assign,
)
from lexsubgen.datasets.base_reader import DatasetReader
from lexsubgen.evaluations.task import Task
from lexsubgen.utils.params import Params, build_from_params
from lexsubgen.utils.register import memory


class WebRunner:
    app = None

    def __init__(
        self, config_path: str, create_generator: bool = False, flask_name: str = None
    ):
        self.app = Flask(flask_name or __name__)
        CORS(self.app)

        self.config_path = config_path
        config_path = Path(config_path)
        if config_path.suffix == ".jsonnet":
            config = json.loads(evaluate_file(str(config_path)))
        elif config_path.suffix == ".json":
            with open(config_path, "r") as fp:
                config = json.load(fp)
        else:
            raise ValueError(
                "Configuration files should be provided in json or jsonnet format"
            )
        params = Params(config)
        self.verbose = params.pop("verbose")
        self.host, self.port = config.pop("host"), config.pop("port")
        self.port = check_and_assign(self.port, self.verbose)
        self.server_name = params.pop("name")

        self.substitutes_cache = cachetools.LRUCache(maxsize=256)

        # Create Substitute Generator from params
        self.generator_params = params.pop("substitute_generator")
        self.substitute_generator = None
        self.task_evaluator = None
        if create_generator:
            self.create_generator()

    def create_generator(self):
        if not self.substitute_generator:
            self.substitute_generator: SubstituteGenerator = build_from_params(
                deepcopy(self.generator_params)
            )
        else:
            print(f" * {self.host}:{self.port} | Substitute generator already created!")
        return {}

    def remove_generator(self):
        if self.substitute_generator:
            del self.substitute_generator
            self.substitute_generator = None
        else:
            print(f" * {self.host}:{self.port} | Substitute generator not created yet!")
        return {}

    def ping(self):
        return {
            "generator_alive": True if self.substitute_generator else False,
        }

    def get_params(self):
        return {"generator_params": self.generator_params.dict}

    def add_endpoint(
        self,
        endpoint_url: str,
        endpoint_name: str,
        handler: Callable,
        method: str = None,
    ) -> NoReturn:
        method = [method.upper()] if method else ["GET"]
        self.app.add_url_rule(
            endpoint_url,
            endpoint_name,
            EndpointAction(handler, endpoint_name, self.server_name),
            methods=method,
        )

    def get_subst(
        self,
        sentences: List[List[str]],
        target_ids: List[int],
        pos_tags: List[str] = None,
        gold_substs: List[List[str]] = None,
        gold_weights: List[List[int]] = None,
        candidates: List[List[str]] = None,
    ):
        # Caching mechanism with LRU cache
        @cachetools.cached(self.substitutes_cache)
        def _get_subst(_sentences, _target_ids):
            if not self.substitute_generator:
                self.create_generator()
            _sentences = [list(sent) for sent in _sentences]
            _generated = self.substitute_generator.generate_substitutes(
                _sentences, list(_target_ids), return_probs=True
            )
            return _generated

        generated_substs, word2id, probs = _get_subst(
            tuple([tuple(sent) for sent in sentences]), tuple(target_ids)
        )
        if gold_substs is None:
            gold_substs = []
        if candidates is None:
            candidates = []
        if gold_weights is None:
            gold_weights = []
        substitutes = [
            {
                "generated": zipped[0],
                "gold": zipped[1],
                "candidates": zipped[2],
                "gold_weights": zipped[3],
            }
            for zipped in zip_longest(
                generated_substs, gold_substs, candidates, gold_weights, fillvalue=None
            )
        ]
        target_words = [
            sentence[target_idx] for sentence, target_idx in zip(sentences, target_ids)
        ]
        return SubstResponse(
            substitutes=substitutes,
            pos_tags=pos_tags,
            target_words=target_words,
            scores=probs,
            word2id=word2id,
        )

    def evaluate(
        self,
        dataset_name: str,
        task_name: str,
        batch_size: int = 50,
        verbose: bool = False,
    ):
        @memory.cache
        def _evaluate(
            _dataset_name, _task_name, _batch_size, _verbose, _generator_params
        ):
            if not self.substitute_generator:
                self.create_generator()

            # TODO: Add pos-tag param logic
            reader_config = {
                "class_name": _task_name + "_reader",
                "dataset_name": _dataset_name,
                "with_pos_tag": True,
            }
            dataset_reader: DatasetReader = build_from_params(Params(reader_config))

            task_config = {
                "class_name": task_name + "_eval",
                "substitute_generator": self.substitute_generator,
                "dataset_reader": dataset_reader,
                "batch_size": _batch_size,
                "verbose": _verbose,
            }
            # noinspection PyTypeHints
            self.task_evaluator: Task = build_from_params(Params(task_config))
            evaluation_result = self.task_evaluator.evaluate()
            _mean_metrics = evaluation_result["mean_metrics"]
            _instance_metrics = evaluation_result["instance_metrics"]
            return _mean_metrics, _instance_metrics

        mean_metrics, instance_metrics = _evaluate(
            dataset_name, task_name, batch_size, verbose, self.generator_params.dict
        )
        # Destruct Task object
        del self.task_evaluator
        self.task_evaluator = None
        return EvalResponse(metrics=mean_metrics)

    def get_progress(self):
        progress = 0
        if self.task_evaluator:
            progress = self.task_evaluator.progress
        return {"progress": int(progress)}

    def run(self, debug: bool = False):
        self.add_endpoint("/ping", "ping", self.ping, method="get")
        self.add_endpoint(
            "/get_progress", "get_progress", self.get_progress, method="get"
        )
        self.add_endpoint("/get_params", "get_params", self.get_params, method="get")

        self.add_endpoint(
            "/create_generator", "create_generator", self.create_generator, method="get"
        )
        self.add_endpoint(
            "/remove_generator", "remove_generator", self.remove_generator, method="get"
        )

        self.add_endpoint(
            "/get_subst",
            "get_subst",
            lambda: self.get_subst(**request.json),
            method="post",
        )
        self.add_endpoint(
            "/evaluate",
            "evaluate",
            lambda: self.evaluate(**request.json),
            method="post",
        )

        self.app.run(host="0.0.0.0", port=self.port, debug=debug)


if __name__ == "__main__":
    fire.Fire(WebRunner)
