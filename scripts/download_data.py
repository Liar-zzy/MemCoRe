"""Download LoCoMo from its official repository."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "locomo10.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    if args.output.exists():
        print(f"Using existing dataset: {args.output}")
        content = args.output.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            content = response.read()
        json.loads(content)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(content)
        print(f"Downloaded: {args.output}")
    samples = json.loads(content)
    questions = sum(len(sample["qa"]) for sample in samples)
    print(f"LoCoMo: {len(samples)} conversations, {questions} questions")


if __name__ == "__main__":
    main()
