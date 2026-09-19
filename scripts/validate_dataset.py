import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arena.dataset import validate_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    summary = {"file": str(args.input), "bytes": args.input.stat().st_size,
               "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               **validate_dataset(args.input)}
    if args.manifest:
        args.manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
