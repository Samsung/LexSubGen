import logging
from pathlib import Path
import json
from typing import Tuple, Dict, Optional, NoReturn, Any, List

import pandas as pd
from fire import Fire
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from overrides import overrides

from lexsubgen.applications.wsi import WSISolver
from lexsubgen.clusterizers.agglo import SubstituteClusterizer
from lexsubgen.datasets.lexsub import DatasetReader
from lexsubgen.evaluations.task import Task
from lexsubgen.hyperparam_search.grid import Grid
from lexsubgen.metrics.wsi_metrics import compute_wsi_metrics
from lexsubgen.runner import Runner
from lexsubgen.subst_generator import SubstituteGenerator
from lexsubgen.utils.mlflow_utils import (
    get_experiment_id,
    log_params,
    log_metrics,
)
from lexsubgen.utils.params import build_from_params, read_config

logger = logging.getLogger(Path(__file__).name)


class WSIEvaluation(Task):
    def __init__(
        self,
        substitute_generator: SubstituteGenerator = None,
        dataset_reader: DatasetReader = None,
        clusterizer: SubstituteClusterizer = None,
        use_pos_tags: bool = True,
        verbose: bool = False,
        batch_size: int = 50,
        results_precision: int = 6,
        save_instance_results: bool = False,
    ):
        """
        Solves the Word Sense Induction task by clustering predicted substitutes
        Args:
            substitute_generator: Object that generate possible substitutes.
            dataset_reader: Object that can read dataset for Lexical Substitution task.
            verbose: Bool flag for verbosity.
            batch_size: Number of samples in batch for substitute generator.
            clusterizer: Object that clusterize generated substitutes.
            results_precision: Final dataframe will be rounded to this value
        """
        super(WSIEvaluation, self).__init__(
            substitute_generator=substitute_generator,
            dataset_reader=dataset_reader,
            verbose=verbose,
        )

        self.wsi_solver = WSISolver(substitute_generator, clusterizer)
        self.results_precision = results_precision
        self.batch_size = batch_size
        self.use_pos_tags = use_pos_tags
        self.save_instance_results = save_instance_results

    @classmethod
    def from_configs(
        cls,
        dataset_reader_config: str,
        substgen_config: str,
        clusterizer_config: str,
        verbose: bool = False,
        batch_size: int = 50,
    ):
        dataset_reader = DatasetReader.from_config(dataset_reader_config)
        substitute_generator = SubstituteGenerator.from_config(
            substgen_config
        )
        clusterizer = SubstituteClusterizer.from_config(
            clusterizer_config
        )
        return cls(
            dataset_reader, substitute_generator, clusterizer, verbose, batch_size
        )

    def dump_metrics(
        self, metrics: Dict[str, Any], run_dir: Optional = None, log: bool = False
    ) -> NoReturn:
        """
        Method for dumping input 'metrics' to 'run_dir' directory.

        Args:
            metrics:
                mean_metrics: Dictionary with mean values of computed metrics
                all_metrics: Dataframe with values for each ambiguous word
            run_dir: Directory path for dumping Lexical Substitution task metrics.
            log: Bool flag for logger.
        """
        mean_metrics = metrics["mean_metrics"]
        all_metrics_df = metrics["all_metrics"]

        if run_dir is not None:
            run_dir = Path(run_dir)
            results_path = run_dir / "results_dataframe.csv"
            all_metrics_df.to_csv(results_path)
            if log: logger.info(f"Results were saved to: {results_path}")

            if "instance_results" in metrics:
                metrics["instance_results"].to_csv(run_dir / "instance_results.csv")

        if log:
            logger.info(f"Mean Metrics: {mean_metrics}")
            logger.info(f"Per Word Metrics:\n{all_metrics_df}")

    def get_instance_results(
        self,
        dataset: Tuple[pd.DataFrame, Dict[str, str], Optional[Path]],
        pred_labels: List[Any],
        idx2substitutes: Dict[int, List[str]],
    ) -> pd.DataFrame:
        """
        Args:
            dataset: Tuple of three objects
                df - pandas dataframe that contains dataset for WSI task
                gold_labels - python dictionary from context id to label
                gold_labels_path - path to file with gold labels (it is necessary
                    to compute metrics for some WSI datasets)
            pred_labels: list of predicted labels

        Returns:
            instance_results: Pandas dataframe with predicted labels and substitutes
        """
        df, gold_labels, _ = dataset

        instance_results = pd.DataFrame()
        instance_results["context_id"] = df["context_id"]
        instance_results["target_id"] = df["target_id"]
        instance_results["target_lemma"] = df["target_lemma"]
        instance_results["pos_tag"] = df["pos_tag"]
        instance_results["gold_label"] = [gold_labels[idx] for idx in df["context_id"]]
        instance_results["pred_label"] = pred_labels
        instance_results["sentence"] = [
            json.dumps([t.upper() if j == tgtid else t for j, t in enumerate(ctx)])
            for _, (ctx, tgtid) in df[["sentence", "target_id"]].iterrows()
        ]
        instance_results["substitutes"] = [
            json.dumps(idx2substitutes[i]) for i in range(len(df))
        ]

        return instance_results

    def get_metrics_from_labels(
        self,
        dataset: Tuple[pd.DataFrame, Dict[str, str], Optional[Path]],
        pred_labels: List[Any],
    ) -> Dict[str, Any]:
        """
        Args:
            dataset: Tuple of three objects
                df - pandas dataframe that contains dataset for WSI task
                gold_labels - python dictionary from context id to label
                gold_labels_path - path to file with gold labels (it is necessary
                    to compute metrics for some WSI datasets)
            pred_labels: list of predicted labels
        Returns:
            mean_metrics: Dictionary with mean values of computed metrics
            all_metrics: Dataframe with values for each ambiguous word
        """
        df, gold_labels, gold_labels_file = dataset

        mean_metrics, all_metrics_df = compute_wsi_metrics(
            y_true=[gold_labels[idx] for idx in df["context_id"]],
            y_pred=pred_labels,
            group_by=df["group_by"].to_list(),
            context_ids=df["context_id"].to_list(),
            y_true_file=gold_labels_file
        )

        all_metrics_df = all_metrics_df.round(self.results_precision)

        return {"mean_metrics": mean_metrics, "all_metrics": all_metrics_df}

    @overrides
    def get_metrics(
        self,
        dataset: Tuple[pd.DataFrame, Dict[str, str], Optional[Path]]
    ) -> Dict[str, Any]:
        """
        Groups @dataset by target word and solves the WSI task for each of them.
        Then calculates the WSI metrics.
        Args:
            dataset: Tuple of three objects
                df - pandas dataframe that contains dataset for WSI task
                gold_labels - python dictionary from context id to label
                gold_labels_path - path to file with gold labels (it is necessary
                    to compute metrics for some WSI datasets)

        Returns:
            mean_metrics: Dictionary with mean values of computed metrics
            all_metrics: Dataframe with values for each ambiguous word
        """
        df, _, _ = dataset

        target_pos_tags = None
        if self.use_pos_tags:
            target_pos_tags = df["pos_tag"].to_list()

        idx2substitutes = self.wsi_solver.substitutes_generation_step(
            tokens_lists=df["sentence"].to_list(),
            target_idxs=df["target_id"].to_list(),
            target_pos=target_pos_tags,
            group_by=df["group_by"].to_list(),
            target_lemmas=df["target_lemma"].to_list(),
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        pred_labels = self.wsi_solver.clustering_step(
            idx2substitutes,
            group_by=df["group_by"].to_list(),
            verbose=self.verbose,
            memory=None,
        )

        metrics = self.get_metrics_from_labels(dataset, pred_labels)
        if self.save_instance_results:
            metrics["instance_results"] = self.get_instance_results(
                dataset, pred_labels, idx2substitutes
            )

        return metrics

    def solve(
        self,
        dataset_config_path: str,
        substgen_config_path: str,
        clusterizer_config_path: str,
        run_dir: str,
        force: bool = False,
        auto_create_subdir: bool = False,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> NoReturn:
        """
        Evaluates task defined by configuration file.
        Builds dataset reader from dataset dataset_config_path and
        substitute generator from substgen_config_path.
        Args:
            config_path: path to a configuration file.
            dataset_config_path: path to a dataset configuration file
            experiment_name: name of the experiment using for MLFlow tracking.
                Default value equal to Task name.
        """
        substgen_config = read_config(substgen_config_path)
        dataset_config = read_config(dataset_config_path)
        clusterizer_config = read_config(clusterizer_config_path)
        config = {
            "class_name": "evaluations.wsi.WSIEvaluation",
            "substitute_generator": substgen_config,
            "dataset_reader": dataset_config,
            "clusterizer": clusterizer_config,
            "verbose": self.verbose,
            "use_pos_tags": self.use_pos_tags,
            "batch_size": self.batch_size,
            "results_precision": self.results_precision,
            "save_instance_results": self.save_instance_results
        }
        runner = Runner(run_dir, force, auto_create_subdir)
        runner.evaluate(config=config, experiment_name=experiment_name, run_name=run_name)

    def hyperparam_search(
        self,
        dataset_config_path: str,
        substgen_config_path: str,
        clusterizer_config_path: str,
        experiment_name: str,
        run_dir: str,
        force: bool = False,
        auto_create_subdir: bool = False,
        dump_hypers_metrics: bool = False,
    ) -> NoReturn:
        """
        Runs hyperparameters enumeration defined by several configuration files.
        Configuration files are given independently because of clustering

        Args:
            dataset_config_path: path to a dataset configuration file.
            substgen_config_path: path to a substitute_generator configuration file.
            clusterizer_config_path: path to a clusterizer configuration file.
            experiment_name: name of the experiment for MLFlow tracking.
            run_dir: path to the directory where to store experiment data.
            force: whether to rewrite data in the existing directory.
            auto_create_subdir: if true a subdirectory will be created automatically
                and its name will be the current date and time
        """
        runner = Runner(run_dir, force, auto_create_subdir)
        self.dataset_reader = self.from_config(dataset_config_path)

        dataset = self.dataset_reader.read_dataset()
        df, _, _ = dataset

        substgen_config = read_config(substgen_config_path, verbose=True)
        clusterizer_config = read_config(clusterizer_config_path, verbose=True)

        run_name = (
            str(Path(dataset_config_path).stem) +
            str(Path(substgen_config_path).stem) +
            str(Path(clusterizer_config_path).stem)
        )

        # Set MLFlow settings
        experiment_id = get_experiment_id(runner.mlflow_client, experiment_name)

        gen_grid = Grid(substgen_config)
        clust_grid = Grid(clusterizer_config)  # TODO: n_cluster must be the last

        target_pos_tags = None
        if self.use_pos_tags:
            target_pos_tags = df["pos_tag"].to_list()

        for i, (gen_grid_dot, gen_config) in enumerate(gen_grid):
            self.wsi_solver.substitute_generator = build_from_params(gen_config)
            idx2substitutes = self.wsi_solver.substitutes_generation_step(
                tokens_lists=df["sentence"].to_list(),
                target_idxs=df["target_id"].to_list(),
                target_pos=target_pos_tags,
                group_by=df["group_by"].to_list(),
                target_lemmas=df["target_lemma"].to_list(),
                batch_size=self.batch_size,
                verbose=self.verbose,
            )

            params_to_log = {k: v for k, v in zip(gen_grid.param_names, gen_grid_dot)}
            for j, (clust_grid_dot, clust_config) in enumerate(clust_grid):
                self.wsi_solver.clusterizer = build_from_params(clust_config)
                pred_labels = self.wsi_solver.clustering_step(
                    idx2substitutes,
                    group_by=df["group_by"].to_list(),
                    verbose=self.verbose,
                    memory=None,
                )

                metrics = self.get_metrics_from_labels(dataset, pred_labels)
                if self.save_instance_results:
                    metrics["instance_results"] = self.get_instance_results(
                        dataset, pred_labels, idx2substitutes
                    )

                if dump_hypers_metrics:
                    hypers_dir = runner.run_dir / f"vectorizer{i}_cluterizer{j}"
                    hypers_dir.mkdir()
                    self.dump_metrics(metrics, hypers_dir, log=False)

                tags = {**runner.git_tags}
                tags[MLFLOW_RUN_NAME] = f"{run_name}_run_{i}_{j}"
                run_entity = runner.mlflow_client.create_run(
                    experiment_id=experiment_id, tags=tags
                )
                params_to_log.update({
                    k: v for k, v in zip(clust_grid.param_names, clust_grid_dot)
                })
                log_params(runner.mlflow_client, run_entity, params_to_log, dict())
                log_metrics(runner.mlflow_client, run_entity, metrics["mean_metrics"])


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(name)-16s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    Fire(WSIEvaluation)
