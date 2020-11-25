from pathlib import Path

from lexsubgen.utils.params import read_config

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "evaluations"


def test_lexsub_evaluation_configs():
    """Checking whether evaluation configs can be loaded or not"""
    for config_path in (CONFIGS_PATH / "lexsub").iterdir():
        read_config(str(config_path))
