import argparse
from pathlib import Path

from app.core.logger import get_logger


log = get_logger("vyasa.download")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw", help="output folder")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Real sources (free):
    # - DharmicData (GitHub)
    # - Vedanta_Datasets (GitHub)
    # - itihasa (GitHub)
    # - GRETIL zip (direct)
    #
    # We download them inside build_corpus.py automatically.
    log.info("Nothing to do here. Use `python scripts\\build_corpus.py` which downloads + builds corpus.")


if __name__ == "__main__":
    main()

