"""Real Qwen3-0.6B decode loop over a multi-session agent workload.

Every configuration runs the same workload (same schedule, same inputs) through
the same paged Qwen3 forward; only the KV memory system changes:

  gpu_only        GPU pool only; evicted blocks are dropped -> recompute (prefill)
  tiered_sync     GPU -> pinned CPU -> SSD, lazy write-back at eviction, every copy
                  blocks the host (no overlap)
  tiered_async    write-through D2H on a side stream as soon as a block fills,
                  FIFO transfer thread, prefetch of the next scheduled session
  tiered_async_sched  as above + scheduler-aware eviction (lookahead 4)

Run through the lock:
  python _tools/gpu_lock.py -- python benchmarks/bench_decode.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kvoffload.manager import BlockManager, Tier  # noqa: E402
from kvoffload.qwen3_paged import PagedQwen3  # noqa: E402
from kvoffload.runtime import AgentWorkload, Runtime  # noqa: E402
from kvoffload.transfer import CudaKVEngine  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
B = 16
CONFIGS = {
    "gpu_only": dict(tiers=False, async_=False, write_through=False, prefetch=False, policy="lru"),
    "tiered_sync": dict(tiers=True, async_=False, write_through=False, prefetch=False, policy="lru"),
    "tiered_async": dict(tiers=True, async_=True, write_through=True, prefetch=True, policy="lru"),
    "tiered_async_sched": dict(tiers=True, async_=True, write_through=True, prefetch=True, policy="sched"),
}


def run_config(name: str, cfg: dict, model: PagedQwen3, wl: AgentWorkload, caps: dict[Tier, int],
               shape: tuple[int, ...]) -> tuple[dict, list[dict], dict[str, list[int]]]:
    caps = caps if cfg["tiers"] else {Tier.GPU: caps[Tier.GPU]}
    eng = CudaKVEngine(shape, caps[Tier.GPU], caps.get(Tier.CPU, 0), caps.get(Tier.SSD, 0),
                       ssd_path=str(ROOT / ".scratch" / f"kv_{name}.bin"), async_=cfg["async_"])
    model.pool = eng.gpu
    mgr = BlockManager(caps, engine=eng, policy=cfg["policy"], write_through=cfg["write_through"])
    rt = Runtime(mgr, B, compute=model, engine=eng, prefetch=cfg["prefetch"])
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    recs = rt.run(wl)
    end.record()
    eng.drain()
    wall = start.elapsed_time(end) / 1e3
    steps = [s for r in recs for s in r.step_ms]
    s = mgr.stats
    hit_blocks = sum(s.hits.values())
    summary = {
        "config": name, "wall_s": round(wall, 2),
        "ttft_ms_mean": round(st.mean(r.ttft_ms for r in recs), 2),
        "ttft_ms_p95": round(float(np.percentile([r.ttft_ms for r in recs], 95)), 2),
        "restore_ms_mean": round(st.mean(r.restore_ms for r in recs), 2),
        "step_ms_p50": round(float(np.percentile(steps, 50)), 3),
        "step_ms_p99": round(float(np.percentile(steps, 99)), 3),
        "host_stall_s": round(eng.stall_s, 3),
        "recomputed_tokens": sum(r.recomputed for r in recs),
        "hit_GPU": s.hits["GPU"], "hit_CPU": s.hits["CPU"], "hit_SSD": s.hits["SSD"],
        "miss_blocks": s.misses,
        "gpu_hit_rate": round(s.hits["GPU"] / max(1, hit_blocks + s.misses), 3),
        "any_tier_hit_rate": round(hit_blocks / max(1, hit_blocks + s.misses), 3),
        "prefetched": s.prefetched, "dropped": s.dropped,
        "copies": dict(s.copies),
        "MB_moved": round(sum(s.copies.values()) * eng.block_bytes / 1e6, 1),
    }
    turns = [{"config": name, "sid": r.sid, "context": r.context, "hit_tokens": r.hit_tokens,
              "recomputed": r.recomputed, "restore_ms": round(r.restore_ms, 3),
              "ttft_ms": round(r.ttft_ms, 3), "step_ms_mean": round(st.mean(r.step_ms), 3)}
             for r in recs]
    outputs = {sid: sess.tokens for sid, sess in rt.sessions.items()}
    eng.close()
    return summary, turns, outputs


def gpu_clocks() -> str:
    """SM clock / power state, recorded because a throttled laptop GPU changes every timing."""
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def save(out: dict, summaries: list[dict], turns: list[dict], outputs: dict, tag: str) -> None:
    """Written after every config, so a run cut short by the time limit still leaves data."""
    ref = outputs.get("tiered_sync") or next(iter(outputs.values()))
    agreement = {}
    for name, o in outputs.items():
        same = sum(sum(a == b for a, b in zip(o[k], ref[k])) for k in ref)
        agreement[name] = round(same / sum(len(ref[k]) for k in ref), 4)
    out.update(token_agreement_vs_tiered_sync=agreement, runs=summaries, clocks_now=gpu_clocks())
    (ROOT / "results" / f"decode{tag}.json").write_text(json.dumps(out, indent=2))
    with open(ROOT / "results" / f"decode{tag}_turns.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(turns[0]))
        w.writeheader()
        w.writerows(turns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-blocks", type=int, default=64)
    ap.add_argument("--cpu-blocks", type=int, default=64)
    ap.add_argument("--ssd-blocks", type=int, default=256)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _main")
    args = ap.parse_args()
    (ROOT / ".scratch").mkdir(exist_ok=True)
    wl = AgentWorkload(n_sessions=args.sessions, n_turns=args.turns,
                       decode_tokens=args.decode_tokens, vocab=150_000)
    caps = {Tier.GPU: args.gpu_blocks, Tier.CPU: args.cpu_blocks, Tier.SSD: args.ssd_blocks}
    shape = PagedQwen3.block_shape(MODEL, B)
    model = PagedQwen3(torch.zeros((8, *shape), dtype=torch.bfloat16, device="cuda"), B, MODEL)
    # Warm-up (CUDA context, cuBLAS heuristics, allocator) on a throwaway workload.
    run_config("warmup", CONFIGS["tiered_async"], model,
               AgentWorkload(n_sessions=2, n_turns=1, decode_tokens=4, vocab=150_000, seed=99), caps, shape)
    out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "model": MODEL,
           "block_size": B, "capacity_blocks": {t.name: c for t, c in caps.items()},
           "block_MiB": round(float(np.prod(shape)) * 2 / 2**20, 3), "workload": wl.__dict__,
           "clocks_start": gpu_clocks()}
    summaries, turns, outputs = [], [], {}
    for rep in range(args.repeats):          # interleave configs across repeats
        for name in args.configs:
            s, t, o = run_config(name, CONFIGS[name], model, wl, caps, shape)
            s["repeat"] = rep
            summaries.append(s)
            turns += [dict(x, repeat=rep) for x in t]
            outputs.setdefault(name, o)
            print(json.dumps(s), flush=True)
            save(out, summaries, turns, outputs, args.tag)
    print(json.dumps(out["token_agreement_vs_tiered_sync"]))
    model.close()


if __name__ == "__main__":
    main()
