import json
import logging
import os
import signal
from pathlib import Path
from typing import List, Optional, Callable, NoReturn, Union

import fire
import requests
from flask import Flask, request, render_template
from flask_cors import CORS

from lexsubgen.analyzer.backend.model_server import WebRunner
from lexsubgen.analyzer.backend.utils import (
    check_and_assign,
    EndpointAction,
    DatasetLoadResponse,
)
from lexsubgen.utils.file import is_valid_jsonnet
from lexsubgen.utils.register import CACHE_DIR, DATASETS_DIR
from lexsubgen.utils.wordnet_relation import Relation

MODEL_CONFIGS_DIRS = CACHE_DIR / "configs" / "web_runners"
SESSION_FILE = CACHE_DIR / "app_session.json"


class WebApp:
    app = None

    def __init__(
        self,
        host: str,
        port: int = 5000,
        model_configs: List[str] = (),
        start_ids: List[int] = (),
        start_all: Optional[bool] = False,
        restore_session: Optional[bool] = False,
    ):
        self.app = Flask(
            "Main App server",
            template_folder=Path(__file__).parent,
            static_folder=Path(__file__).parent / "static",
        )
        CORS(self.app)

        fucking_logger = logging.getLogger(Path(__file__).name)
        fucking_logger.setLevel(logging.ERROR)

        self.host = host
        self.port = check_and_assign(port)
        self.all_configs = list()
        self.all_models = list()
        self.child_processes = list()
        self.parallel_requests_available = False

        if restore_session:
            self.restore_session()
        else:
            self.all_configs = [
                (MODEL_CONFIGS_DIRS / config_path) for config_path in model_configs
            ]

        # Create directory for Custom Datasets if needed
        if not os.path.exists(CACHE_DIR / "analyzer_datasets"):
            os.mkdir(CACHE_DIR / "analyzer_datasets")

        # Create directory for Configs if needed
        if not os.path.exists(MODEL_CONFIGS_DIRS):
            os.mkdir(MODEL_CONFIGS_DIRS)

        if start_ids or start_all:
            self.start_models(start_ids, start_all)

    def restore_session(self):
        with open(SESSION_FILE, "r") as fp:
            session_info = json.load(fp)
        self.parallel_requests_available = session_info["is_parallel"]
        for model in session_info["models_info"]:
            config_path = model["config_path"]
            self.add_model(config_path, pad_prefix=False, is_active=model["is_active"])

    @staticmethod
    def index():
        return {"index": render_template("index.html")}

    def start_models(self, start_ids: List[int], start_all: bool) -> NoReturn:
        for idx in start_ids:
            # Check for proper ids
            assert idx >= len(self.all_configs)
        if start_all:
            start_ids = list(range(len(self.all_configs)))

        for config_path in self.all_configs:
            self.add_model(config_path, pad_prefix=False, check=False)

        # Run models specified in start_ids
        for idx in start_ids:
            host = self.all_models[idx]["host"]
            port = self.all_models[idx]["port"]
            requests.get(f"http://{host}:{port}/create_generator")

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
            EndpointAction(handler, endpoint_name, "Main App"),
            methods=method,
        )

    def get_models(self):
        response = {str(idx): model for idx, model in enumerate(self.all_models)}
        response["is_parallel"] = self.parallel_requests_available
        return response

    def switch_parallel(self):
        self.parallel_requests_available = not self.parallel_requests_available
        return {"is_parallel": self.parallel_requests_available}

    def change_status(self, model_idx):
        # Check model_idx for valid value
        assert model_idx in list(range(len(self.all_models)))

        model_info = self.all_models[model_idx]
        self.all_models[model_idx]["is_active"] = not model_info["is_active"]
        # If user deactivate model, then server needed to free memory
        if not self.all_models[model_idx]["is_active"]:
            host = self.all_models[model_idx]["host"]
            port = self.all_models[model_idx]["port"]
            requests.get(f"http://{host}:{port}/remove_generator")

        response = {"changed_model_info": self.all_models[model_idx]}
        return response

    def remove_model(self, model_idx):
        # Check model_idx for valid value
        assert model_idx in list(range(len(self.all_models)))

        removed_model_info = self.all_models.pop(model_idx)
        removed_config_path = self.all_configs.pop(model_idx)
        pid = self.child_processes.pop(model_idx)
        os.kill(pid, signal.SIGKILL)
        response = {"pid": pid, "removed_model_info": removed_model_info}
        return response

    def add_model(
        self,
        config_path: Union[str, Path],
        pad_prefix: bool = True,
        check: bool = True,
        is_active: bool = True,
    ):
        config_path = Path(config_path)
        if pad_prefix:
            config_path = MODEL_CONFIGS_DIRS / config_path

        if check:
            config_index = (
                self.all_configs.index(config_path)
                if config_path in self.all_configs
                else -1
            )
            if config_index != -1 and len(self.all_models) < config_index:
                print(f" * {self.host}:{self.port} Config already exist: {config_path}")
                return {"added_model_info": self.all_models[config_index]}
            else:
                if config_path.suffix == ".jsonnet":
                    if not is_valid_jsonnet(config_path):
                        raise ValueError("Invalid jsonnet config!")
                self.all_configs.append(config_path)
        model_server = WebRunner(config_path)
        self.all_models.append(
            {
                "config_path": str(config_path),
                "is_active": is_active,
                "host": model_server.host,
                "port": model_server.port,
                "name": model_server.server_name,
            }
        )
        ppid = os.fork()
        if ppid == 0:
            # Child process
            model_server.run()
            exit(0)
        self.child_processes.append(ppid)
        return {"added_model_info": self.all_models[-1]}

    @staticmethod
    def upload_dataset():
        dataset_filename: str = request.form["datasetName"]
        if not dataset_filename.endswith(".json"):
            raise ValueError("Invalid file extension. Dataset must be in JSON format!")
        dataset_content = request.form["datasetContent"]
        datasets_path = CACHE_DIR / "analyzer_datasets" / dataset_filename
        with open(datasets_path, "w") as fp:
            fp.write(dataset_content)
        return {"dataset_path": str(datasets_path)}

    @staticmethod
    def upload_generator_config():
        config_name = Path(request.form["configName"])
        if not (config_name.suffix == ".json" or config_name.suffix == ".jsonnet"):
            raise ValueError(
                "Invalid config file extension. Config must be in JSON or Jsonnet format!"
            )
        config_content = request.form["configContent"]
        config_path = MODEL_CONFIGS_DIRS / config_name
        with open(config_path, "w") as fp:
            fp.write(config_content)
        return {"config_path": str(config_path)}

    @staticmethod
    def dataset_names(is_custom: bool = False):
        if is_custom:
            datasets = os.listdir(CACHE_DIR / "analyzer_datasets")
            datasets = [f[:-5] for f in datasets if f.endswith(".json")]
        else:
            datasets = os.listdir(DATASETS_DIR)
        return {"dataset_names": datasets}

    @staticmethod
    def config_names():
        configs = os.listdir(MODEL_CONFIGS_DIRS)
        return {"config_names": configs}

    @staticmethod
    def wordnet_relations():
        possible_wordnet_relations = [relation.name for relation in Relation]
        return {"relations": possible_wordnet_relations}

    def load_dataset(self, dataset_name: str):
        existing_datasets = self.dataset_names(is_custom=True)["dataset_names"]
        if dataset_name not in existing_datasets:
            raise ValueError(f"Dataset with name '{dataset_name}' doesn't exist!")
        with open(
            CACHE_DIR / "analyzer_datasets" / (dataset_name + ".json"), "r"
        ) as fp:
            dataset = json.load(fp)
        return DatasetLoadResponse(samples=dataset)

    def run(self, debug: bool = False):
        """
        Main function to start Web Application
        :param debug: bool flag, if debug=True server runs in debug mode,
            i.e. server will restart when you change code of application
        :return:
        """
        print(f"Running Application on port: {self.port}")

        # Entry point route
        self.add_endpoint("/", "index", self.index, method="get")

        # Model management routes
        self.add_endpoint("/get_models", "get_models", self.get_models, method="get")
        self.add_endpoint(
            "/switch_parallel", "switch_parallel", self.switch_parallel, method="get"
        )
        self.add_endpoint(
            "/change_status",
            "change_status",
            lambda: self.change_status(**request.json),
            method="post",
        )
        self.add_endpoint(
            "/remove_model",
            "remove_model",
            lambda: self.remove_model(**request.json),
            method="post",
        )
        self.add_endpoint(
            "/add_model",
            "add_model",
            lambda: self.add_model(**request.json),
            method="post",
        )

        # Custom datasets management routes
        self.add_endpoint(
            "/upload_dataset", "upload_dataset", self.upload_dataset, method="post"
        )
        self.add_endpoint(
            "/dataset_names",
            "dataset_names",
            lambda: self.dataset_names(**request.json),
            method="post",
        )
        self.add_endpoint(
            "/load_dataset",
            "load_dataset",
            lambda: self.load_dataset(**request.json),
            method="post",
        )

        # Configs management routes
        self.add_endpoint(
            "/upload_config",
            "upload_config",
            self.upload_generator_config,
            method="post",
        )
        self.add_endpoint(
            "/config_names", "config_names", self.config_names, method="get"
        )

        # Wordnet relation route
        self.add_endpoint(
            "/wordnet_relations",
            "wordnet_relations",
            self.wordnet_relations,
            method="get",
        )

        self.app.run(host="0.0.0.0", port=self.port, debug=debug)
        with open(SESSION_FILE, "w") as fp:
            session_info = {
                "is_parallel": self.parallel_requests_available,
                "models_info": [
                    {
                        "config_path": model["config_path"],
                        "is_active": model["is_active"],
                    }
                    for model in self.all_models
                ],
            }
            json.dump(session_info, fp, indent=4)
        print(f"Server off: session dumped to {SESSION_FILE}")
        for pid in self.child_processes:
            os.system(f"kill -9 {pid}")


def main():
    fire.Fire(WebApp)


if __name__ == "__main__":
    main()
