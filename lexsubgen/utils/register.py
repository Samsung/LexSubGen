import logging
import os
from pathlib import Path

from joblib import Memory

logger = logging.getLogger(Path(__file__).name)
logger.setLevel(logging.INFO)

CACHE_DIR = Path(os.environ["HOME"]) / ".cache" / "lexsubgen"
DATASETS_DIR = CACHE_DIR / "datasets"
ENTRY_DIR = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = ENTRY_DIR / "configs"
LOGS_DIR = ENTRY_DIR / "logs"
FRONTEND_DIR = ENTRY_DIR / "lex"
MEMORY_CACHE_PATH = CACHE_DIR / "function_cache"
memory = Memory(str(MEMORY_CACHE_PATH), verbose=0)
