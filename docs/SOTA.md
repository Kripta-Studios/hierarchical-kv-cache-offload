# State of the art: hierarchical KV-cache storage (as of late 2026)

This is a short, opinionated map of the work this repo borrows from. Every entry
links to a source I actually read while building the project; numbers are the
authors' own claims, not something reproduced here.

## 1. Paging and prefix reuse on the GPU

| Work | Idea we reuse |
|---|---|
| **PagedAttention / vLLM** - Kwon et al., SOSP'23, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | KV cache stored in fixed-size, non-contiguous *blocks* addressed through a block table, like OS virtual memory pages. The authors report near-zero KV memory waste and 2-4x throughput over earlier systems. Our `BlockManager` is a paged allocator in exactly this sense. |
| **vLLM Automatic Prefix Caching** - [design doc](https://docs.vllm.ai/en/latest/design/v1/prefix_caching/), [RFC #2614](https://github.com/vllm-project/vllm/issues/2614) | A block is identified by `hash(parent_hash, tokens_in_block, extras)`; only *full* blocks are cached; freed cached blocks sit in an LRU queue and are only really evicted when the slot is needed. The docs say the default hash is SHA-256 since v0.11, and a per-request *cache salt* in the first block isolates tenants. Our `hashing.py` follows that scheme. |
| **SGLang RadixAttention** - Zheng et al., [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) | Same goal, different structure: the KV of all requests is a radix tree keyed by token prefix, which handles reuse at arbitrary boundaries (not only block-aligned). |

## 2. Moving KV below the GPU: CPU RAM, local SSD, remote storage

| Work | What it does | Relevance |
|---|---|---|
| **CachedAttention (ex-AttentionStore)** - Gao et al., USENIX ATC'24, [arXiv:2403.19708](https://arxiv.org/abs/2403.19708) | Keeps the KV of *inactive conversation sessions* in host memory and disks instead of discarding it between turns. It overlaps transfers with compute (layer-wise pre-loading, asynchronous saving) and places/evicts KV using hints from the job scheduler. Authors report up to 87% lower TTFT and up to 70% lower end-to-end cost. | The closest match to this repo: multi-turn sessions, a 3-tier hierarchy, and scheduler-aware eviction (`SchedulerAwarePolicy`). |
| **Mooncake** (Kimi / Moonshot AI) - Qin et al., [arXiv:2407.00079](https://arxiv.org/abs/2407.00079), [code](https://github.com/kvcache-ai/Mooncake) | A KVCache-centric *disaggregated* architecture: separate prefill and decode clusters, with a distributed KV store built from the GPU servers' otherwise idle CPU, DRAM, SSD and NICs, plus a global scheduler (Conductor) that balances cache reuse against SLOs. | Shows where the idea goes at cluster scale: the KV pool becomes a service of its own. |
| **LMCache** - Liu et al., [arXiv:2510.09665](https://arxiv.org/abs/2510.09665), [code](https://github.com/LMCache/LMCache) | A KV-cache layer that sits beside vLLM/SGLang and stores KV in CPU memory, local disk, remote disk or Redis, moving it over Ethernet/RDMA/NVLink. Authors report up to 15x throughput on multi-round QA and document analysis. | The production version of "offload KV from the engine and share it". |
| **SSD offloading in LMCache** - [arXiv:2610.21249](https://arxiv.org/abs/2610.21249) | Characterises predictability vs. latency trade-offs of KV SSD offloading in LMCache. | A reminder that the SSD tier adds tail latency. That is also what we see on this laptop (direct-I/O read of one block takes about 0.9 ms). |
| **vLLM native KV Offloading Connector** - [blog (IBM Research, Jan 2026)](https://vllm.ai/blog/kv-offloading-connector), [usage guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/) | Native CPU offloading since vLLM 0.11, built on the asynchronous connector API added in 0.9. Copies use `cudaMemcpyAsync` (DMA engine), not a copy kernel. The key lesson is that the old per-layer K/V layout produced tiny transfer units (for example 32 KB for Llama-3.1-8B). Storing all layers of a block contiguously made blocks 2 MB and raised offload throughput by about an order of magnitude. On H100 they measured 83.4 GB/s bidirectional DMA with 2 MB blocks. Later docs list `lru` and `arc` eviction policies. | We use the same **all-layers-contiguous block layout** (1.75 MiB per block for Qwen3-0.6B) and DMA copies from pinned memory. Our pinned-vs-size measurements show why the layout matters. |
| **NVIDIA Dynamo KV Block Manager (KVBM)** - [design docs](https://docs.nvidia.com/dynamo/latest/architecture/kvbm_components.html) | Names the tiers G1 (device), G2 (pinned host), G3 (local SSD) and G4 (remote/network storage). Blocks are offloaded down the tiers and onboarded back up, with NIXL doing the transfers, and the docs describe a block lifecycle (Reset, Partial, Complete, Registered). | Same G1->G2->G3 cascade as our `Tier` enum. Their "Partial -> Complete -> Registered" lifecycle is our "unhashed -> full -> committed" block. |

## 3. Offloading for a single long sequence (not multi-session)

| Work | Idea |
|---|---|
| **InfiniGen** - Lee et al., OSDI'24, [arXiv:2406.19707](https://arxiv.org/abs/2406.19707) | During decode it *speculates* at layer i-1 which tokens will matter at layer i, and prefetches only those K/V entries from CPU memory. The authors report up to 3x speedup over earlier KV management methods. This is a smarter prefetch than ours, which works per session. |
| **KVSwap** - [arXiv:2511.11907](https://arxiv.org/abs/2511.11907) | Disk-aware KV offloading for long-context inference *on device*. |

## 4. Where this repo sits

We implement, from scratch and at small scale, the control plane plus data plane
that all of the above share:

* a paged block pool per tier with an inclusive GPU -> pinned CPU -> SSD hierarchy,
* prefix-chained SHA-256 block identity with reference counting (vLLM APC),
* LRU, and scheduler-aware eviction (CachedAttention-style lookahead),
* write-through async offload on a side CUDA stream, plus prefetch of the next
  scheduled session, with event-based hazard tracking,
* all-layers-contiguous blocks, which turn each transfer into one large DMA (the
  vLLM connector lesson).

What we leave out: layer-wise pipelining of loads (CachedAttention, LMCache),
GPUDirect Storage / NIXL / RDMA, KV compression or quantization for the lower
tiers, distributed and disaggregated pools (Mooncake), radix-tree reuse at
arbitrary boundaries (SGLang), and token-level selective prefetch (InfiniGen).
