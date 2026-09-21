import argparse
import csv
import json
import statistics
from pathlib import Path


def load(spec: str) -> tuple[str, list[dict]]:
    label, filename = spec.split("=", 1)
    filename, _, agent = filename.partition("#")
    with Path(filename).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return label, [row for row in rows if not agent or row["agent"] == agent]


def summarize(label: str, rows: list[dict]) -> dict:
    number = lambda key: [float(row.get(key, 0)) for row in rows]
    return {
        "label": label,
        "episodes": len(rows),
        "mean_reward": statistics.mean(number("reward")),
        "median_reward": statistics.median(number("reward")),
        "mean_ticks": statistics.mean(number("ticks")),
        "gems": int(sum(number("gems"))),
        "kills": int(sum(number("kills"))),
        "damage": int(sum(number("damage"))),
        "deaths": int(sum(number("death"))),
        "win_rate": statistics.mean(number("win")),
        "mean_hp_remaining": statistics.mean(number("hp_remaining")),
        "environment_kills": int(sum(number("environment_kills"))),
        "mean_branch_factor": statistics.mean(number("mean_branch_factor")),
        "deadlock_rate": statistics.mean(number("deadlock_rate")),
        "unique_action_rate": statistics.mean(number("unique_action_rate")),
        "mean_entropy": statistics.mean(number("mean_entropy")),
        "skill_efficiency": statistics.mean(number("skill_efficiency")),
        "mean_latency_p95_ms": statistics.mean(number("latency_p95_ms")),
    }


def svg(summaries: list[dict], path: Path) -> None:
    width, height, left, top = 820, 420, 170, 55
    maximum = max(item["mean_reward"] for item in summaries) or 1
    palette = ("#4d8cf5", "#4ade80", "#fbbf24", "#f87171", "#a78bfa")
    rows = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#0c111c"/>',
            '<text x="24" y="30" fill="#e8eef7" font-family="sans-serif" font-size="20" font-weight="700">Jev Arena · Mean Reward</text>']
    for index, item in enumerate(summaries):
        y = top + index * 65
        bar = round((width - left - 70) * item["mean_reward"] / maximum)
        rows += [f'<text x="24" y="{y + 23}" fill="#aab7ca" font-family="sans-serif" font-size="15">{item["label"]}</text>',
                 f'<rect x="{left}" y="{y}" width="{bar}" height="34" rx="5" fill="{palette[index % len(palette)]}"/>',
                 f'<text x="{left + bar + 10}" y="{y + 23}" fill="#e8eef7" font-family="sans-serif" font-size="15">{item["mean_reward"]:.2f}</text>']
    rows.append('</svg>')
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", required=True, help="LABEL=CSV[#AGENT]")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark_report"))
    args = parser.parse_args()
    summaries = [summarize(*load(spec)) for spec in args.series]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    header = "| Agent | Episodes | Mean reward | Median | Gems | Kills | Damage | Deaths | P95 ms |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    lines = [f"| {x['label']} | {x['episodes']} | {x['mean_reward']:.2f} | {x['median_reward']:.2f} | {x['gems']} | {x['kills']} | {x['damage']} | {x['deaths']} | {x['mean_latency_p95_ms']:.1f} |" for x in summaries]
    (args.output_dir / "summary.md").write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    svg(summaries, args.output_dir / "reward.svg")
    print(json.dumps(summaries))


if __name__ == "__main__":
    main()
