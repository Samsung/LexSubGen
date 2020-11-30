import json
import logging
from pathlib import Path
from typing import Dict, List, Any, NoReturn, Optional

import numpy as np
import pandas as pd
from fire import Fire
from overrides import overrides
from collections import OrderedDict
from tqdm import tqdm

from lexsubgen.datasets.lexsub import DatasetReader
from lexsubgen.evaluations.task import Task
from lexsubgen.metrics.all_word_ranking_metrics import (
    compute_precision_recall_f1_topk,
    compute_precision_recall_f1_vocab
)
from lexsubgen.metrics.candidate_ranking_metrics import gap_score
from lexsubgen.runner import Runner
from lexsubgen.subst_generator import SubstituteGenerator
from lexsubgen.utils.batch_reader import BatchReader
from lexsubgen.utils.file import dump_json
from lexsubgen.utils.params import read_config
from lexsubgen.utils.wordnet_relation import to_wordnet_pos, get_wordnet_relation

logger = logging.getLogger(Path(__file__).name)

DEFAULT_RUN_DIR = Path(__file__).resolve().parent.parent.parent / "debug" / Path(__file__).stem


class LexSubEvaluation(Task):
    def __init__(
        self,
        substitute_generator: SubstituteGenerator = None,
        dataset_reader: DatasetReader = None,
        verbose: bool = True,
        k_list: List[int] = (1, 3, 10),
        batch_size: int = 50,
        save_instance_results: bool = True,
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
        self.save_instance_results = save_instance_results

        self.gap_metrics = ["gap", "gap_normalized", "gap_vocab_normalized"]
        self.base_metrics = ["precision", "recall", "f1_score"]
        k_metrics = []
        for k in self.k_list:
            k_metrics.extend([f"prec@{k}", f"rec@{k}", f"f1@{k}"])
        self.metrics = self.gap_metrics + self.base_metrics + k_metrics

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

        logger.info(f"Lexical Substitution for {len(dataset)} instances.")

        progress_bar = BatchReader(
            dataset["context"].tolist(),
            dataset["target_position"].tolist(),
            dataset["pos_tag"].tolist(),
            dataset["gold_subst"].tolist(),
            dataset["gold_subst_weights"].tolist(),
            dataset["candidates"].tolist(),
            dataset["target_lemma"].tolist(),
            batch_size=self.batch_size,
        )

        if self.verbose:
            progress_bar = tqdm(
                progress_bar,
                desc=f"Lexical Substitution for {len(dataset)} instances"
            )

        all_metrics_data, columns = [], None

        for (
            tokens_lists,
            target_ids,
            pos_tags,
            gold_substitutes,
            gold_weights,
            candidates,
            target_lemmas,
        ) in progress_bar:

            # Computing probability distribution over possible substitutes
            probs, word2id = self.substitute_generator.get_probs(
                tokens_lists, target_ids, pos_tags
            )

            # Selecting most probable substitutes from the obtained distribution
            pred_substitutes = self.substitute_generator.substitutes_from_probs(
                probs, word2id, tokens_lists, target_ids
            )

            # Ranking candidates using the obtained distribution
            ranked = self.substitute_generator.candidates_from_probs(
                probs, word2id, candidates
            )
            ranked_candidates_in_vocab, ranked_candidates = ranked

            for i in range(len(pred_substitutes)):
                instance_results = OrderedDict([
                    ("target_word", tokens_lists[i][target_ids[i]]),
                    ("target_lemma", target_lemmas[i]),
                    ("target_pos_tag", pos_tags[i]),
                    ("target_position", target_ids[i]),
                    ("context", json.dumps(tokens_lists[i])),
                ])

                # Metrics computation
                # Compute GAP, GAP_normalized, GAP_vocab_normalized and ranked candidates
                gap_scores = gap_score(
                    gold_substitutes[i], gold_weights[i],
                    ranked_candidates_in_vocab[i], word2id,
                )
                for metric, gap in zip(self.gap_metrics, gap_scores):
                    instance_results[metric] = gap

                # Computing basic Precision, Recall, F-score metrics
                base_metrics_values = compute_precision_recall_f1_vocab(
                    gold_substitutes[i], word2id
                )
                for metric, value in zip(self.base_metrics, base_metrics_values):
                    instance_results[metric] = value

                # Computing Top K metrics for each K in the k_list
                k_metrics = compute_precision_recall_f1_topk(
                    gold_substitutes[i], pred_substitutes[i], self.k_list
                )
                for metric, value in k_metrics.items():
                    instance_results[metric] = value

                if self.save_instance_results:
                    additional_results = self.create_instance_results(
                        tokens_lists[i], target_ids[i], pos_tags[i],
                        probs[i], word2id, gold_weights[i],
                        gold_substitutes[i], pred_substitutes[i],
                        candidates[i], ranked_candidates[i]
                    )
                    instance_results.update(
                        (k, v) for k, v in additional_results.items()
                    )

                all_metrics_data.append(list(instance_results.values()))

                if columns is None:
                    columns = list(instance_results.keys())

        all_metrics = pd.DataFrame(all_metrics_data, columns=columns)

        mean_metrics = {
            metric: round(all_metrics[metric].mean(skipna=True) * 100, 2)
            for metric in self.metrics
        }

        return {"mean_metrics": mean_metrics, "instance_metrics": all_metrics}

    def create_instance_results(
        self,
        tokens: List[str], target_id: int, pos_tag: str, probs: np.ndarray,
        word2id: Dict[str, int], gold_weights: Dict[str, int],
        gold_substitutes: List[str], pred_substitutes: List[str],
        candidates: List[str], ranked_candidates: List[str],
    ) -> Dict[str, Any]:
        instance_results = OrderedDict()
        pos_tag = to_wordnet_pos.get(pos_tag, None)
        target = tokens[target_id]
        instance_results["gold_substitutes"] = json.dumps(gold_substitutes)
        instance_results["gold_weights"] = json.dumps(gold_weights)
        instance_results["pred_substitutes"] = json.dumps(pred_substitutes)
        instance_results["candidates"] = json.dumps(candidates)
        instance_results["ranked_candidates"] = json.dumps(ranked_candidates)

        if hasattr(self.substitute_generator, "prob_estimator"):
            prob_estimator = self.substitute_generator.prob_estimator
            if target in word2id:
                instance_results["target_subtokens"] = 1
            elif hasattr(prob_estimator, "tokenizer"):
                target_subtokens = prob_estimator.tokenizer.tokenize(target)
                instance_results["target_subtokens"] = len(target_subtokens)
            else:
                instance_results["target_subtokens"] = -1

        if self.save_target_rank:
            target_rank = -1
            if target in word2id:
                target_vocab_idx = word2id[target]
                target_rank = np.where(np.argsort(-probs) == target_vocab_idx)[0][0]
            instance_results["target_rank"] = target_rank

        if self.save_wordnet_relations:
            relations = [
                get_wordnet_relation(target, s, pos_tag)
                for s in pred_substitutes
            ]
            instance_results["relations"] = json.dumps(relations)

        return instance_results

    @overrides
    def dump_metrics(
        self, metrics: Dict[str, Any], run_dir: Path, log: bool = False
    ):
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
            if self.save_instance_results:
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
            "save_instance_results": self.save_instance_results,
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
