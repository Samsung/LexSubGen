from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import git
import pkg_resources
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT, MLFLOW_GIT_BRANCH

from lexsubgen.utils.register import ENTRY_DIR


def log_params(client: MlflowClient, run: Run, params: Dict, accumulated_params: Dict):
    for key, value in params.items():
        if key in ["class_name", "verbose", "batch_size", "cuda_device"]:
            continue
        if isinstance(value, list):
            [log_params(client, run, item, accumulated_params) for item in value]
        elif isinstance(value, dict):
            log_params(client, run, value, accumulated_params)
        else:
            key_idx = 0
            param_name = key
            while param_name in accumulated_params.keys():
                key_idx += 1
                param_name = f"{key}_{key_idx}"
            accumulated_params[param_name] = value
            client.log_param(run_id=run.info.run_uuid, key=param_name, value=value)


def log_metrics(
    client: MlflowClient, run: Run, metrics: Dict, step: Optional[int] = None
):
    for key, value in metrics.items():
        key = key.replace("@", "_")
        client.log_metric(run_id=run.info.run_uuid, key=key, value=value, step=step)


def get_experiment_id(client: MlflowClient, experiment_name: str) -> int:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def get_git_info(path: Union[str, Path]) -> Optional[Tuple[str, str]]:
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        path = path.parent
    try:
        repo = git.Repo(path)
        commit = repo.head.commit.hexsha
        branch = repo.active_branch.name
        return commit, branch
    except (
        git.InvalidGitRepositoryError,
        git.GitCommandNotFound,
        ValueError,
        git.NoSuchPathError,
    ):
        return None


def get_git_tags() -> Optional[Dict]:
    git_info = get_git_info(ENTRY_DIR)
    tags = dict()
    if git_info is not None:
        for key, value in zip([MLFLOW_GIT_COMMIT, MLFLOW_GIT_BRANCH], git_info):
            tags[key] = value
    return tags


def get_lib_versions() -> Dict[str, str]:
    lib_versions = dict()
    for pkg in pkg_resources.working_set:
        lib_versions[pkg.project_name] = pkg.version
    return lib_versions
