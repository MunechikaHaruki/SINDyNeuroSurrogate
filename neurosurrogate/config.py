from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SURROGATE_DATA_DIR = DATA_DIR / "surrogate"
PYX_DATA_DIR = DATA_DIR / "pyx"

data_dir_list = [
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    SURROGATE_DATA_DIR,
    PYX_DATA_DIR,
]

for l in data_dir_list:
    l.mkdir(parents=True, exist_ok=True)

for d in [INTERIM_DATA_DIR, RAW_DATA_DIR]:
    for t in ["hh", "hh3", "traub"]:
        target_path = d / t
        target_path.mkdir(parents=True, exist_ok=True)


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
