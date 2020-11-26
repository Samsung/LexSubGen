import os
from pathlib import Path
from typing import List
import pytest

from lexsubgen.utils.params import read_config

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs"

def get_configs_paths(path: str) -> List[str]:
    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            config_path = os.path.join(root, file)
            if config_path.endswith(".jsonnet"):
                paths.append(config_path)
    return paths


configs = get_configs_paths(str(CONFIGS_PATH))


@pytest.mark.parametrize("config_path", configs)
def test_loading_all_configs(config_path: str):
    read_config(str(config_path))
