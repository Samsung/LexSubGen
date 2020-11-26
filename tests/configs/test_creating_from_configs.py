import os
import pytest
from pathlib import Path
from typing import List

from lexsubgen.utils.params import build_from_config_path

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


def get_configs_paths(path: str) -> List[str]:
    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            config_path = os.path.join(root, file)
            if config_path.endswith(".jsonnet") and "hyperparam_search" not in config_path:
                paths.append(config_path)
    return paths


configs = get_configs_paths(str(CONFIGS_PATH))


@pytest.mark.parametrize("config_path", configs)
def test_loading_all_configs(config_path: str):
    print(f"Trying to load '{config_path}' config")
    build_from_config_path(str(config_path))
