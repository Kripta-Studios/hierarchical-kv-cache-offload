"""Hit-rate simulation (no GPU): the same BlockManager + Runtime, with a NullEngine.

Sweeps capacity for a larger agent workload than the real decode benchmark
(32 sessions x 12 turns) and compares LRU with the scheduler-aware policy.
Outputs results/policy_sweep.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kvoffload.manager import BlockManager, Tier  # noqa: E402
from kvoffload.runtime import AgentWorkload, Runtime  # noqa: E402

B = 16


def simulate(caps: dict[Tier, int], policy: str, wl: AgentWorkload) -> dict:
    mgr = BlockManager(caps, policy=policy, write_through=True)
    rt = Runtime(mgr, B, prefetch=False)
    rt.run(wl)
    s = mgr.stats
    total = sum(s.hits.values()) + s.misses
    return {"gpu_hit": s.hits["GPU"] / total, "cpu_hit": s.hits["CPU"] / total,
            "ssd_hit": s.hits["SSD"] / total, "miss": s.misses / total,
            "recomputed_tokens": sum(r.recomputed for r in rt.records)}


def main() -> None:
    wl = AgentWorkload(n_sessions=32, n_turns=12, decode_tokens=32)
    rows = []
    # (a) GPU-only: how much recompute does each policy save as VRAM grows?
    # The longest session (~2.9k tokens = ~182 blocks) must fit on the GPU while it runs.
    for gpu in (256, 384, 512, 768, 1024, 1536, 2048, 3072):
        for policy in ("lru", "sched"):
            r = simulate({Tier.GPU: gpu}, policy, wl)
            rows.append({"experiment": "gpu_only", "gpu_blocks": gpu, "cpu_blocks": 0,
                         "ssd_blocks": 0, "policy": policy, **r})
    # (b) Tiered: fixed small GPU, growing pinned CPU, big SSD behind it.
    for cpu in (0, 256, 512, 1024, 2048):
        for policy in ("lru", "sched"):
            caps = {Tier.GPU: 256, Tier.CPU: cpu, Tier.SSD: 4096 if cpu else 0}
            r = simulate(caps, policy, wl)
            rows.append({"experiment": "tiered", "gpu_blocks": 256, "cpu_blocks": cpu,
                         "ssd_blocks": caps[Tier.SSD], "policy": policy, **r})
    for r in rows:
        for k in ("gpu_hit", "cpu_hit", "ssd_hit", "miss"):
            r[k] = round(r[k], 4)
        print(r)
    with open(ROOT / "results" / "policy_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
