import logging
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import List, Optional, NoReturn, Dict

import fire
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from tqdm import tqdm

from lexsubgen.hyperparam_search.grid import Grid
from lexsubgen.utils.file import create_run_dir, import_submodules, dump_json
from lexsubgen.utils.mlflow_utils import (
    get_experiment_id,
    get_git_tags,
    log_params,
    log_metrics,
    get_lib_versions,
)
from lexsubgen.utils.params import (
    build_from_config_path, build_from_params, read_config
)
from lexsubgen.utils.register import ENTRY_DIR

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)


class Runner:
    @staticmethod
    def import_additional_modules(additional_modules):
        # Import additional modules
        logger.info("Importing additional modules...")
        if additional_modules is not None:
            if not isinstance(additional_modules, list):
                additional_modules = [additional_modules]
            for additional_module in additional_modules:
                import_submodules(additional_module)

    def __init__(self, run_dir: str, force: bool = False, auto_create_subdir: bool = False):
        """
        Class that handles command line interaction with the LexSubGen framework.
        Different methods of this class are related to different scenarios of framework usage.
        E.g. evaluate method performs substitute generator evaluation on the dataset specified
        in the configuration.

        Args:
            run_dir: path to the directory where to store experiment data.
            force: whether to rewrite data in the existing directory.
            auto_create_subdir: if true a subdirectory will be created automatically
                and its name will be the current date and time
        """
        self.run_dir = Path(run_dir)
        if auto_create_subdir and not force:
            time_str = datetime.now().isoformat().split('.')[0]
            self.run_dir = self.run_dir / f"{time_str}"
        self.force = force
        self.git_tags = get_git_tags()
        self.lib_versions = get_lib_versions()

        # Create run directory
        logger.info(f"Creating run directory {self.run_dir}...")
        create_run_dir(self.run_dir, force=self.force)
        dump_json(self.run_dir / "lib_versions.json", self.lib_versions)

        self.mlflow_dir = str(ENTRY_DIR / "mlruns")
        mlflow.set_tracking_uri(self.mlflow_dir)
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_dir)

    def evaluate(
        self,
        config_path: str = None,
        config: Optional[Dict] = None,
        additional_modules: Optional[List[str]] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> NoReturn:
        """
        Evaluates task defined by configuration file.

        Args:
            config_path: path to a configuration file.
            config: configuration of a task.
            additional_modules: path to directories with modules that should be registered in global Registry.
            experiment_name: results of the run will be added to 'experiment_name' experiment in MLflow.
            run_name: this run will be marked as 'run_name' in MLflow.
        """
        # Instantiate objects from config
        task, config = build_from_config_path(config_path, config)

        self.import_additional_modules(additional_modules)

        # Create experiment with given name or get already existing
        if experiment_name is None:
            experiment_name = config["class_name"]
        experiment_id = get_experiment_id(self.mlflow_client, experiment_name)

        # Add Run name in MLFlow tags
        tags = copy(self.git_tags)
        if config_path is not None and run_name is None:
            run_name = Path(config_path).stem
        if run_name is not None:
            tags[MLFLOW_RUN_NAME] = run_name

        # Create Run entity for tracking
        run_entity = self.mlflow_client.create_run(
            experiment_id=experiment_id, tags=tags
        )
        saved_params = dict()
        generator_params = config["substitute_generator"]
        log_params(self.mlflow_client, run_entity, generator_params, saved_params)

        dump_json(self.run_dir / "config.json", config)

        logger.info("Evaluating...")
        metrics = task.evaluate(run_dir=self.run_dir)
        metrics = metrics["mean_metrics"]
        log_metrics(self.mlflow_client, run_entity, metrics)
        self.mlflow_client.log_artifacts(
            run_entity.info.run_uuid, local_dir=self.run_dir
        )
        logger.info("Evaluation performed.")

    def hyperparam_search(self, config_path: str, experiment_name: str) -> NoReturn:
        """
        Run hyperparameters enumeration defined by configuration file.

        Args:
            config_path: path to a configuration file.
            experiment_name: results of the run will be added to 'experiment_name' experiment in MLflow.
        """
        config = read_config(config_path, verbose=True)

        run_name = Path(config_path).stem

        dump_json(self.run_dir / "config.json", config)

        # Set MLFlow settings
        experiment_id = get_experiment_id(self.mlflow_client, experiment_name)

        parameter_grid = Grid(config)
        for run_idx, (grid_dot, param_config) in tqdm(enumerate(parameter_grid)):
            tags = copy(self.git_tags)
            tags[MLFLOW_RUN_NAME] = f"{run_name}_run_{run_idx}"
            run_entity = self.mlflow_client.create_run(
                experiment_id=experiment_id, tags=tags
            )
            params_to_log = {k: v for k, v in zip(parameter_grid.param_names, grid_dot)}
            log_params(self.mlflow_client, run_entity, params_to_log, dict())
            task = build_from_params(param_config)
            metrics = task.evaluate(run_dir=self.run_dir)
            metrics = metrics.get("mean_metrics", None)
            log_metrics(self.mlflow_client, run_entity, metrics)

    def augment(
        self, dataset_name: str, config_path: str = None, config: Optional[Dict] = None
    ):
        """
        Performs dataset augmentation.

        Args:
            dataset_name: name of the dataset to augment
            config_path: path to a configuration file.
            config: configuration of a task
        """
        augmenter, config = build_from_config_path(config_path, config)

        dump_json(self.run_dir / "config.json", config)

        logger.info(f"Augmenting {dataset_name}...")
        augmented_dataset = augmenter.augment_dataset(dataset_name=dataset_name)
        augmented_dataset.to_csv(
            self.run_dir / "augmented_dataset.tsv", sep="\t", index=False
        )
        logger.info(f"Augmentation performed. Results was saved in {self.run_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(name)-16s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # This is necessary to use fire as an entry_point
    fire.Fire(Runner)
