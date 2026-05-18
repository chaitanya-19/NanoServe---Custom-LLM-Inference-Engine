"""Compare two benchmark runs and emit a markdown table + plots.

Usage:
    python -m benchmark.compare \\
        --inputs results_nanoserve.json results_vllm.json \\
        --output-md comparison.md \\
        --output-dir plots/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _make_table(runs: List[dict]) -> str:
    # One row per (engine, concurrency); columns are metrics
    headers = [
        "engine", "conc",
        "req/s", "tok/s",
        "TTFT P50 (ms)", "TTFT P95 (ms)", "TTFT P99 (ms)",
        "ITL P50 (ms)", "ITL P95 (ms)", "ITL P99 (ms)",
        "E2E P50 (s)", "E2E P95 (s)", "E2E P99 (s)",
        "ok/total",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for run in runs:
        engine = run["engine_name"]
        for r in run["results"]:
            row = [
                engine,
                str(r["concurrency"]),
                f"{r['requests_per_sec']:.2f}",
                f"{r['output_tokens_per_sec']:.1f}",
                f"{r['ttft_p50']*1000:.0f}",
                f"{r['ttft_p95']*1000:.0f}",
                f"{r['ttft_p99']*1000:.0f}",
                f"{r['itl_p50']*1000:.1f}",
                f"{r['itl_p95']*1000:.1f}",
                f"{r['itl_p99']*1000:.1f}",
                f"{r['e2e_p50']:.2f}",
                f"{r['e2e_p95']:.2f}",
                f"{r['e2e_p99']:.2f}",
                f"{r['num_success']}/{r['num_requests']}",
            ]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _make_plots(runs: List[dict], out_dir: Path) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    metric_specs = [
        ("output_tokens_per_sec", "Output tokens / sec", "throughput.png"),
        ("requests_per_sec", "Requests / sec", "rps.png"),
        ("ttft_p50", "TTFT P50 (s)", "ttft_p50.png"),
        ("ttft_p95", "TTFT P95 (s)", "ttft_p95.png"),
        ("itl_p50", "ITL P50 (s)", "itl_p50.png"),
        ("itl_p95", "ITL P95 (s)", "itl_p95.png"),
    ]
    for key, label, fname in metric_specs:
        plt.figure(figsize=(7, 4.5))
        for run in runs:
            xs = [r["concurrency"] for r in run["results"]]
            ys = [r[key] for r in run["results"]]
            plt.plot(xs, ys, marker="o", label=run["engine_name"])
        plt.xlabel("concurrency")
        plt.ylabel(label)
        plt.title(label + " vs concurrency")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = out_dir / fname
        plt.savefig(path, dpi=120)
        plt.close()
        files.append(str(path))
    return files


def main() -> None:
    p = argparse.ArgumentParser("nanoserve-compare")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output-md", default="comparison.md")
    p.add_argument("--output-dir", default="plots")
    args = p.parse_args()

    runs = [_load(x) for x in args.inputs]
    md = _make_table(runs)
    files = _make_plots(runs, Path(args.output_dir))

    with open(args.output_md, "w") as f:
        f.write("# NanoServe vs baseline\n\n")
        f.write(f"Models: {', '.join(set(r['model'] for r in runs))}\n\n")
        f.write("## Aggregate metrics\n\n")
        f.write(md + "\n\n")
        if files:
            f.write("## Plots\n\n")
            for fp in files:
                rel = Path(fp).name
                f.write(f"![{rel}]({Path(args.output_dir).name}/{rel})\n\n")

    print(f"wrote {args.output_md}")
    for fp in files:
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
