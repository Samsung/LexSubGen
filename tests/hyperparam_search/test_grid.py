import json
from _jsonnet import evaluate_file
from collections import Iterable

from lexsubgen.utils.register import CONFIGS_DIR
from lexsubgen.hyperparam_search.grid import Grid


def test_grid():
    config_path = CONFIGS_DIR / "hyperparam_search" / "megatron.jsonnet"
    config = json.loads(evaluate_file(str(config_path)))
    grid = Grid(config)
    assert isinstance(grid.param_names, Iterable)
    assert isinstance(grid.param_values, Iterable)
