import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, NoReturn, Optional

import numpy as np
import pandas as pd
from fire import Fire
from overrides import overrides
from tqdm import tqdm

from lexsubgen.datasets.lexsub import DatasetReader
from lexsubgen.evaluations.task import Task
from lexsubgen.metrics.all_word_ranking_metrics import precision_recall_f1_score
from lexsubgen.metrics.candidate_ranking_metrics import gap_score
from lexsubgen.runner import Runner
from lexsubgen.subst_generator import SubstituteGenerator
from lexsubgen.utils.batch_reader import BatchReader
from lexsubgen.utils.file import dump_json
from lexsubgen.utils.params import read_config
from lexsubgen.utils.wordnet_relation import to_wordnet_pos, get_wordnet_relation

logger = logging.getLogger(Path(__file__).name)

NSUBSTS_TO_SAVE = 200
DEFAULT_RUN_DIR = Path(__file__).resolve().parent.parent.parent / "debug" / Path(__file__).stem


class LexSubEvaluation(Task):
    def __init__(
        self,
        substitute_generator: SubstituteGenerator = None,
        dataset_reader: DatasetReader = None,
        verbose: bool = True,
        k_list: List[int] = (1, 3, 10),
        batch_size: int = 50,
        save_dataset_info: bool = False,
        save_wordnet_relations: bool = False,
        save_target_rank: bool = False,
    ):
        """
        Main class for performing Lexical Substitution task evaluation.
        This evaluation computes metrics for two subtasks in Lexical Substitution task:

        - Candidate-ranking task (GAP, GAP_normalized, GAP_vocab_normalized).
        - All-word-ranking task (Precision@k, Recall@k, F1@k for k-best substitutes).

        Args:
            substitute_generator: Object that generate possible substitutes.
            dataset_reader: Object that can read dataset for Lexical Substitution task.
            verbose: Bool flag for verbosity.
            k_list: List of integer numbers for metrics. For example, if 'k_list' equal to [1, 3, 5],
                then there will calculating the following metrics:
                    - Precion@1, Recall@1, F1-score@1
                    - Precion@3, Recall@3, F1-score@3
                    - Precion@5, Recall@5, F1-score@5
            batch_size: Number of samples in batch for substitute generator.
        """
        super(LexSubEvaluation, self).__init__(
            substitute_generator=substitute_generator,
            dataset_reader=dataset_reader,
            verbose=verbose,
        )
        self.batch_size = batch_size
        self.k_list = k_list
        self.save_wordnet_relations = save_wordnet_relations
        self.save_target_rank = save_target_rank
        self.save_dataset_info = save_dataset_info
        # logger.setLevel(logging.DEBUG if not verbose else logging.INFO)
        # output_handler = logging.StreamHandler(sys.stdout)
        # formatter = logging.Formatter(
        #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        # )
        # output_handler.setFormatter(formatter)
        # logger.addHandler(output_handler)

    @classmethod
    def from_configs(
        cls,
        substgen_config_path: str,
        dataset_config_path: str,
        *args, **kwargs
    ):
        dataset = DatasetReader.from_config(dataset_config_path)
        substgen = SubstituteGenerator.from_config(substgen_config_path)
        return LexSubEvaluation(substgen, dataset, *args, **kwargs)

    @overrides
    def get_metrics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Method for calculating metrics for Lexical Substitution task.

        Args:
            dataset: pandas DataFrame with the whole dataset.
        Returns:
            metrics_data: Dictionary with two keys:

                - all_metrics: pandas DataFrame, extended 'dataset' with computed metrics
                - mean_metrics: Dictionary with mean values of computed metrics
        """
        gap_metrics = ["gap", "gap_normalized", "gap_vocab_normalized"]
        base_metrics = ["precision", "recall", "f1_score"]
        k_metrics = []
        for k in self.k_list:
            k_metrics.extend([f"prec@{k}", f"rec@{k}", f"f1@{k}"])

        metrics = gap_metrics + base_metrics + k_metrics
        logger.info(
            f"Lexical Substitution for {len(dataset)} instances. "
            f"Computing {len(metrics)} metrics: {', '.join([m.title() for m in metrics])}"
        )

        dataframe_cols = [
            "context",
            "target_word",
            "gold_subst",
            "gold_weights",
            "candidates",
            "ranked_candidates",
            "pred_subst",
        ]
        dataframe_cols += metrics
        all_metrics = pd.DataFrame(columns=dataframe_cols)

        progress_bar = BatchReader(
            dataset["context"].tolist(),
            dataset["target_position"].tolist(),
            dataset["pos_tag"].tolist(),
            dataset["gold_subst"].tolist(),
            dataset["gold_subst_weights"].tolist(),
            dataset["candidates"].tolist(),
            batch_size=self.batch_size,
        )

        if self.verbose:
            progress_bar = tqdm(progress_bar)

        for (
            tokens_lists,
            target_ids,
            pos_tags,
            gold_subst,
            gold_w,
            cands,
        ) in progress_bar:

            batch_prediction, word2id = self.substitute_generator.get_probs(
                tokens_lists, target_ids, pos_tags
            )
            vocabulary = list(word2id.keys())
            id2word = {idx: word for word, idx in word2id.items()}
            batch_size, vocab_size = batch_prediction.shape
            for i in range(batch_size):
                instance_results = dict()
                prediction = batch_prediction[i, :]
                gold_substitutes = gold_subst[i]
                gold_weights = gold_w[i]
                candidates = cands[i]

                # Metrics computation
                # Compute GAP, GAP_normalized, GAP_vocab_normalized and ranked candidates
                gap_scores = gap_score(
                    gold_substitutes,
                    gold_weights,
                    prediction,
                    word2id,
                    candidates,
                    return_ranked_candidates=True,
                )
                for metric, gap in zip(gap_metrics, gap_scores):
                    if gap is not None:
                        instance_results[metric] = gap[0]
                    else:
                        instance_results[metric] = None
                if gap_scores[0] is not None:
                    instance_results["ranked_candidates"] = gap_scores[0][1]

                # Compute basic Precision, Recall, F-score metrics and K metrics for each K in give k_list
                *base_metrics_values, k_metrics = precision_recall_f1_score(
                    gold_substitutes, prediction, word2id, self.k_list
                )
                for metric, value in zip(base_metrics, base_metrics_values):
                    instance_results[metric] = value
                for k, values in k_metrics.items():
                    for metric, value in zip(["prec", "rec", "f1"], values):
                        instance_results[f"{metric}@{k}"] = value

                if self.save_dataset_info:
                    pos_tag = to_wordnet_pos.get(pos_tags[i], None)
                    context = tokens_lists[i]
                    target = context[target_ids[i]]
                    instance_results["context"] = json.dumps(context)
                    instance_results["target_word"] = target
                    instance_results["target_position"] = target_ids[i]
                    instance_results["target_pos_tag"] = pos_tag
                    instance_results["gold_subst"] = json.dumps(gold_substitutes)
                    instance_results["gold_weights"] = json.dumps(gold_weights)
                    instance_results["candidates"] = json.dumps(candidates)
                    n = len(prediction)
                    parted_idxs = np.argpartition(
                        prediction, kth=range(-min(NSUBSTS_TO_SAVE, n), 0)
                    )
                    sorted_top_idxs = parted_idxs[-min(NSUBSTS_TO_SAVE, n):]
                    substitutes = [id2word[idx] for idx in sorted_top_idxs][::-1]
                    instance_results["pred_subst"] = json.dumps(substitutes)
                    probs = [
                        float(prediction[idx])
                        for idx in sorted_top_idxs
                    ][::-1]
                    instance_results["pred_probs"] = json.dumps(probs)

                    prob_estimator = self.substitute_generator.prob_estimator
                    if target in word2id:
                        instance_results["target_subtokens"] = 1
                    elif hasattr(prob_estimator, "tokenizer"):
                        target_subtokens = prob_estimator.tokenizer.tokenize(
                            target
                        )
                        instance_results["target_subtokens"] = len(
                            target_subtokens
                        )
                    else:
                        instance_results["target_subtokens"] = -1

                    if self.save_target_rank:
                        if target in word2id:
                            target_vocab_idx = word2id[target]
                            target_rank = np.where(
                                np.argsort(-prediction) == target_vocab_idx
                            )[0][0]
                        else:
                            target_rank = -1
                        instance_results["target_rank"] = target_rank

                    if self.save_wordnet_relations:
                        relations = [
                            get_wordnet_relation(target, s, pos_tag)
                            for s in substitutes
                        ]
                        instance_results["relations"] = json.dumps(relations)

                all_metrics = all_metrics.append(instance_results, ignore_index=True)

            if self.verbose:
                new_description = "|"
                mean_current_gap = round(
                    all_metrics["gap_normalized"].mean(skipna=True) * 100, 2
                )
                new_description += "GAP: " + str(mean_current_gap)
                if "prec@1" in metrics:
                    mean_current_prec_at_one = round(
                        all_metrics["prec@1"].mean(skipna=True) * 100, 2
                    )
                    new_description += " P@1: " + str(mean_current_prec_at_one)
                if "prec@3" in metrics:
                    mean_current_prec_at_three = round(
                        all_metrics["prec@3"].mean(skipna=True) * 100, 2
                    )
                    new_description += " P@3: " + str(mean_current_prec_at_three)
                if "rec@10" in metrics:
                    mean_current_rec_at_ten = round(
                        all_metrics["rec@10"].mean(skipna=True) * 100, 2
                    )
                    new_description += " R@10: " + str(mean_current_rec_at_ten)
                new_description += "|"
                progress_bar.set_description(desc=new_description)

        mean_metrics = dict()
        for metric in metrics:
            # logger.debug(f"{metric.title()} support: {all_metrics[metric].count()}")
            # logger.debug(
            #     f"{metric.title()} nan values: {all_metrics[metric].isna().sum()}"
            # )
            mean_metrics[metric] = round(all_metrics[metric].mean(skipna=True) * 100, 2)

        metrics_data = {"mean_metrics": mean_metrics, "instance_metrics": all_metrics}
        return metrics_data

    @overrides
    def dump_metrics(
        self, metrics: Dict[str, Any], run_dir: Path, log: bool = False
    ) -> NoReturn:
        """
        Method for dumping input 'metrics' to 'run_dir' directory.

        Args:
            metrics: Dictionary with two keys:

                - all_metrics: pandas DataFrame, extended 'dataset' with computed metrics
                - mean_metrics: Dictionary with mean values of computed metrics
            run_dir: Directory path for dumping Lexical Substitution task metrics.
            log: Bool flag for logger.
        """
        if run_dir is not None:
            with (run_dir / "metrics.json").open("w") as fp:
                json.dump(metrics["mean_metrics"], fp, indent=4)
            metrics_df: pd.DataFrame = metrics["instance_metrics"]
            metrics_df.to_csv(run_dir / "results.csv", sep=",", index=False)
            metrics_df.to_html(run_dir / "results.html", index=False)
            if log:
                logger.info(f"Evaluation results were saved to '{run_dir.resolve()}'")
        if log:
            logger.info(json.dumps(metrics["mean_metrics"], indent=4))

    def solve(
        self,
        substgen_config_path: str,
        dataset_config_path: str,
        run_dir: str = DEFAULT_RUN_DIR,
        mode: str = "evaluate",
        force: bool = False,
        auto_create_subdir: bool = True,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> NoReturn:
        """
        Evaluates task defined by configuration files.
        Builds dataset reader from dataset dataset_config_path and
        substitute generator from substgen_config_path.

        Args:
            substgen_config_path: path to a configuration file.
            dataset_config_path: path to a dataset configuration file.
            run_dir: path to the directory where to store experiment data.
            mode: evaluation mode - 'evaluate' or 'hyperparam_search'
            force: whether to rewrite data in the existing directory.
            auto_create_subdir: if true a subdirectory will be created automatically
                and its name will be the current date and time
            experiment_name: results of the run will be added to 'experiment_name' experiment in MLflow.
            run_name: this run will be marked as 'run_name' in MLflow.
        """
        substgen_config = read_config(substgen_config_path)
        dataset_config = read_config(dataset_config_path)
        config = {
            "class_name": "evaluations.lexsub.LexSubEvaluation",
            "substitute_generator": substgen_config,
            "dataset_reader": dataset_config,
            "verbose": self.verbose,
            "k_list": self.k_list,
            "batch_size": self.batch_size,
            "save_wordnet_relations": self.save_wordnet_relations,
            "save_target_rank": self.save_target_rank,
        }
        runner = Runner(run_dir, force, auto_create_subdir)
        dump_json(Path(run_dir) / "config.json", config)
        if mode == "evaluate":
            runner.evaluate(
                config=config,
                experiment_name=experiment_name,
                run_name=run_name
            )
        elif mode == "hyperparam_search":
            runner.hyperparam_search(
                config_path=Path(run_dir) / "config.json",
                experiment_name=experiment_name
            )


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(name)-16s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    Fire(LexSubEvaluation)
