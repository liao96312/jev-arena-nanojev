import argparse
import json
import math
from pathlib import Path


def probabilities(logits: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = math.fsum(weights)
    return [value / total for value in weights]


def metrics(rows: list[dict], temperature: float) -> dict:
    ce = kl = tv = brier = 0.0
    for row in rows:
        target = row["gold_distribution_probs"]
        predicted = probabilities(row["student_logits"], temperature)
        ce += -math.fsum(q * math.log(max(p, 1e-12)) for q, p in zip(target, predicted))
        kl += math.fsum(q * math.log(q / max(p, 1e-12)) for q, p in zip(target, predicted) if q)
        tv += .5 * math.fsum(abs(q - p) for q, p in zip(target, predicted))
        brier += math.fsum((q - p) ** 2 for q, p in zip(target, predicted))
    return {"questions": len(rows), "target_ce": ce / len(rows), "target_kl": kl / len(rows),
            "target_tv": tv / len(rows), "target_brier": brier / len(rows)}


def fit_temperature(rows: list[dict]) -> tuple[float, bool]:
    objective = lambda log_t: metrics(rows, math.exp(log_t))["target_ce"]
    lo, hi = -4.0, 4.0
    ratio = (math.sqrt(5) - 1) / 2
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    left_value, right_value = objective(left), objective(right)
    for _ in range(80):
        if left_value < right_value:
            hi, right, right_value = right, left, left_value
            left = hi - ratio * (hi - lo)
            left_value = objective(left)
        else:
            lo, left, left_value = left, right, right_value
            right = lo + ratio * (hi - lo)
            right_value = objective(right)
    temperature = math.exp((lo + hi) / 2)
    return (temperature, True) if objective(math.log(temperature)) < objective(0.0) else (1.0, False)


def read_predictions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty predictions: {path}")
    for row in rows:
        logits, target = row.get("student_logits"), row.get("gold_distribution_probs")
        if (not isinstance(logits, list) or not isinstance(target, list) or len(logits) != len(target) or
                len(logits) < 2 or any(not math.isfinite(value) for value in logits + target) or
                any(value < 0 for value in target) or not math.isclose(math.fsum(target), 1, abs_tol=1e-6)):
            raise ValueError(f"invalid prediction row: {row.get('id')}")
    return rows


def calibrate(run_dir: Path) -> dict:
    splits = {name: read_predictions(run_dir / f"predictions_{name}.jsonl")
              for name in ("dev", "calibration", "test", "ood")}
    temperature, fitted = fit_temperature(splits["calibration"])
    return {
        "method": "global scalar temperature fitted on held-out calibration policy targets",
        "interpretation": "action-policy distribution recovery; not empirical event-probability calibration",
        "temperature": temperature,
        "temperature_fitted": fitted,
        "temperature_search_log_bounds": [-4.0, 4.0],
        "temperature_search_boundary_hit": temperature >= math.exp(4.0) * (1 - 1e-9),
        "selected_on": "calibration target CE",
        "metrics_by_split": {
            name: {"before": metrics(rows, 1.0), "after": metrics(rows, temperature)}
            for name, rows in splits.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = calibrate(args.run_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
