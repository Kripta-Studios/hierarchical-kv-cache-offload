"""Turn results/*.json|csv into the PNGs embedded in the README (CPU only)."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
COLORS = {"lru": "#4C72B0", "sched": "#DD8452"}


def policy_sweep() -> None:
    rows = list(csv.DictReader(open(RES / "policy_sweep.csv")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for pol in ("lru", "sched"):
        g = [r for r in rows if r["experiment"] == "gpu_only" and r["policy"] == pol]
        a.plot([int(r["gpu_blocks"]) for r in g], [1 - float(r["miss"]) for r in g],
               "o-", color=COLORS[pol], label=pol)
        t = [r for r in rows if r["experiment"] == "tiered" and r["policy"] == pol]
        b.plot([int(r["cpu_blocks"]) for r in t], [1 - float(r["miss"]) for r in t],
               "o-", color=COLORS[pol], label=f"{pol}: any tier")
        b.plot([int(r["cpu_blocks"]) for r in t], [float(r["gpu_hit"]) for r in t],
               "--", color=COLORS[pol], alpha=0.6, label=f"{pol}: GPU only")
    a.set(title="GPU-only pool (32 sessions x 12 turns)", xlabel="GPU blocks (16 tokens each)",
          ylabel="KV block hit rate", ylim=(0, 1))
    b.set(title="256 GPU blocks + pinned CPU + 4096 SSD blocks", xlabel="pinned CPU blocks",
          ylabel="KV block hit rate", ylim=(0, 1))
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "policy_sweep.png", dpi=110)


def decode() -> None:
    by = defaultdict(list)
    for p in sorted(RES.glob("decode*.json")):   # one file per bench_decode.py --tag
        for r in json.loads(p.read_text())["runs"]:
            by[r["config"]].append(r)
    if not by:
        return
    names = list(by)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    med = lambda k: [sorted(x[k] for x in by[n])[len(by[n]) // 2] for n in names]  # noqa: E731
    a.bar(names, med("restore_ms_mean"), color="#4C72B0")
    a.set(title="Mean KV restore time per turn (before prefill)", ylabel="ms")
    x = range(len(names))
    b.bar([i - 0.2 for i in x], med("step_ms_p50"), 0.4, label="p50", color="#4C72B0")
    b.bar([i + 0.2 for i in x], med("step_ms_p99"), 0.4, label="p99", color="#DD8452")
    b.set_xticks(list(x), names)
    b.set(title="Decode step latency (GPU shared with other jobs)", ylabel="ms")
    b.legend()
    for ax in (a, b):
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "decode.png", dpi=110)


if __name__ == "__main__":
    policy_sweep()
    decode()
