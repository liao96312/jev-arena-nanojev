import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "NanoJev" / "scripts"))

from predict_toy_decisions import prepare_examples
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--tokenizer", type=Path,
                        default=ROOT / "runs" / "arena_rollout_memory_head_50step" / "tokenizer")
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                               trust_remote_code=False)
    lengths = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        payload = {"states": [{key: row[key] for key in ("id", "state", "questions")}]}
        lengths.extend(max(map(len, example["leaf_tokens"]))
                       for example in prepare_examples(payload, tokenizer, args.max_length))
    if not lengths:
        raise ValueError("dataset contains no questions")
    lengths.sort()
    percentile = lambda fraction: lengths[round((len(lengths) - 1) * fraction)]
    print(json.dumps({"questions": len(lengths), "p50": percentile(.5), "p95": percentile(.95),
                      "max": lengths[-1], "max_length": args.max_length}))


if __name__ == "__main__":
    main()
