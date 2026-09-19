import json
import math
from pathlib import Path


SPLITS = ("train",) * 14 + ("dev",) * 2 + ("calibration", "test", "test", "ood")


def split_for_seed(seed: int) -> str:
    return SPLITS[seed % len(SPLITS)]


def validate_dataset(path: str | Path) -> dict:
    counts: dict[str, int] = {}
    groups: dict[str, str] = {}
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "state_id", "family_id", "split", "state", "questions",
                        "gold_probs", "gold_probs_kind", "metadata"}
            if not required <= set(row):
                raise ValueError(f"line {line_number}: missing required fields")
            if row["id"] in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id")
            seen_ids.add(row["id"])
            split = row["split"]
            if split not in set(SPLITS):
                raise ValueError(f"line {line_number}: invalid split")
            group = row["metadata"].get("source_group_id")
            if not isinstance(group, str) or not group:
                raise ValueError(f"line {line_number}: missing source group")
            if group in groups and groups[group] != split:
                raise ValueError(f"line {line_number}: source group leaks across splits")
            groups[group] = split
            question = row["questions"].get("action", {})
            candidates = question.get("criteria", {})
            probabilities = row["gold_probs"].get("action", {})
            if set(candidates) != set(probabilities):
                raise ValueError(f"line {line_number}: candidate/probability keys differ")
            if len(candidates) < 2:
                raise ValueError(f"line {line_number}: choice requires at least two candidates")
            if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities.values()):
                raise ValueError(f"line {line_number}: invalid probability")
            if not math.isclose(math.fsum(probabilities.values()), 1, abs_tol=1e-9):
                raise ValueError(f"line {line_number}: probabilities do not sum to one")
            counts[split] = counts.get(split, 0) + 1
    if not seen_ids:
        raise ValueError("dataset is empty")
    return {"records": len(seen_ids), "groups": len(groups), "splits": counts}
