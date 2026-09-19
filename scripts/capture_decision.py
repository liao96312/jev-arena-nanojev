import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arena import ArenaEnv
from nanojev_adapter.client import NanoJevClient
from nanojev_adapter.schema import decision_request


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/validation/decision_response.json"))
    args = parser.parse_args()
    env, client = ArenaEnv(), NanoJevClient()
    env.reset(args.seed)
    request = decision_request(env)
    record = {"health": client.health(), "request": request, "response": client.evaluate(request)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
