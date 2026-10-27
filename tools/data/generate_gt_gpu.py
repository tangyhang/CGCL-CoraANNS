#!/usr/bin/env python3
"""Stream fbin vectors through an exhaustive GPU L2 search; write ivecs GT.

FP32 GEMM is only a candidate filter. A conservative rounding-error guard
retains every possible top-k candidate, including ties. Selected distances are
recomputed as sum((float64(base) - float64(query))**2), then merged on CPU.
No ANN index, normalization, TF32, FP16, or fixed-size reranking shortlist is
used. See GENERATE_LAION_GT.md for the error bound and compatibility policy.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time

import numpy as np
import torch


def read_fbin(path: Path) -> np.memmap:
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"Truncated fbin header: {path}")
    rows, dim = struct.unpack("<II", header)
    if not rows or not dim or path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError(f"Invalid fbin dimensions or file size: {path}")
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def read_ivecs(path: Path, nq: int, nb: int) -> np.ndarray:
    if path.stat().st_size % 4:
        raise ValueError(f"Invalid ivecs byte count: {path}")
    raw = np.fromfile(path, dtype="<i4")
    k = int(raw[0]) if raw.size else 0
    if k <= 0 or raw.size != nq * (k + 1):
        raise ValueError(f"Invalid ivecs layout or query count: {path}")
    records = raw.reshape(nq, k + 1)
    if not np.all(records[:, 0] == k):
        raise ValueError(f"Inconsistent ivecs row lengths: {path}")
    ids = records[:, 1:].astype(np.int64)
    if np.any(ids < 0) or np.any(ids >= nb):
        raise ValueError(f"Out-of-range GT IDs: {path}")
    if any(len(np.unique(row)) != k for row in ids):
        raise ValueError(f"Duplicate IDs in an existing GT row: {path}")
    return ids


def error_guard(qnorm64: np.ndarray, max_bnorm32: float, dim: int) -> np.ndarray:
    """Absolute error bound for FP32 ||q||² + ||b||² - 2 q.b.

    gamma_(2d+16) covers products, reductions, and the two final additions.
    Cauchy-Schwarz bounds sum(abs(q_i*b_i)). The factor two gives slack for
    the bound's own evaluation and FP64 refinement roundoff. A tiny absolute
    term covers subnormal products, including hardware flush-to-zero.
    """
    unit = np.finfo(np.float32).eps / 2
    operations = 2 * dim + 16
    if operations * unit >= 0.01:
        raise ValueError("Dimension too large for the FP32 error guard")
    gamma = operations * unit / (1 - operations * unit)
    bnorm_upper = max_bnorm32 / (1 - gamma)
    scale = qnorm64 + bnorm_upper + 2 * np.sqrt(qnorm64 * bnorm_upper)
    return 2 * gamma * scale + 16 * dim * np.finfo(np.float32).tiny


def iter_base_blocks(base: np.ndarray, block_rows: int, device: torch.device):
    """Double-buffer sequential reads and, on CUDA, overlap H2D with compute."""
    capacity = min(block_rows, len(base))
    buffers = [
        torch.empty((capacity, base.shape[1]), dtype=torch.float32,
                    pin_memory=device.type == "cuda")
        for _ in range(2)
    ]
    transfer = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    last_copy = [None, None]

    def stage(start: int, slot: int):
        if last_copy[slot] is not None:
            last_copy[slot].synchronize()  # DMA must finish before reusing host memory.
        count = min(capacity, len(base) - start)
        host = buffers[slot][:count]
        np.copyto(host.numpy(), base[start:start + count])
        if transfer is None:
            return start, host, None
        with torch.cuda.device(device), torch.cuda.stream(transfer):
            gpu = host.to(device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(transfer)
            last_copy[slot] = event
        return start, gpu, event

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(stage, 0, 0)
        for block, start in enumerate(range(0, len(base), capacity)):
            begin, vectors, event = pending.result()
            if event is not None:
                current = torch.cuda.current_stream(device)
                current.wait_event(event)
                vectors.record_stream(current)
            next_start = start + capacity
            if next_start < len(base):
                pending = pool.submit(stage, next_start, (block + 1) % 2)
            yield begin, vectors


def merge_topk(old_ids, old_distances, new_ids, new_distances, k):
    ids = np.concatenate((old_ids, new_ids))
    distances = np.concatenate((old_distances, new_distances))
    # Blocks contain disjoint base IDs, so the union has no duplicate IDs.
    order = np.lexsort((ids, distances))[:k]
    return ids[order], distances[order]


@torch.inference_mode()
def exhaustive_l2(
    base: np.ndarray, queries: np.ndarray, *, k: int = 200,
    device: str = "cuda:0", base_block: int = 262144,
    query_batch: int = 1024, refine_batch: int = 32768, progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if base.ndim != 2 or queries.ndim != 2 or base.shape[1] != queries.shape[1]:
        raise ValueError("Base and query dimensions must match")
    if not 0 < k <= len(base) or not len(queries):
        raise ValueError("Require 0 < topk <= base rows and at least one query")
    if min(base_block, query_batch, refine_batch) < 1:
        raise ValueError("Block sizes must be positive")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use a CUDA device, or cpu for small correctness tests")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this script does not silently use CPU")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE to ensure FP32 GEMM")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    qhost = np.array(queries, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(qhost).all():
        raise ValueError("Queries contain NaN or infinity")
    q64host = qhost.astype(np.float64)
    qnorm64 = np.einsum("ij,ij->i", q64host, q64host)
    # Keep every FP32 intermediate comfortably below overflow.
    norm_limit = np.finfo(np.float32).max / 16
    if qnorm64.max() >= norm_limit:
        raise ValueError("Query norms are too large for safe FP32 filtering")
    qgpu = torch.from_numpy(qhost).to(device)
    q64gpu = torch.from_numpy(q64host).to(device)
    qnorm32 = qgpu.square().sum(dim=1)
    nq, dim = queries.shape
    best_ids = [np.empty(0, dtype=np.int64) for _ in range(nq)]
    best_distances = [np.empty(0, dtype=np.float64) for _ in range(nq)]
    cutoffs = np.full(nq, np.inf, dtype=np.float64)
    refined = 0
    started = time.monotonic()

    for begin, bgpu in iter_base_blocks(base, base_block, device):
        bnorm32 = bgpu.square().sum(dim=1)
        # Amax propagates NaN; norms also expose infinities in the input.
        max_norm = float(bnorm32.amax().item())
        if not math.isfinite(max_norm) or max_norm >= norm_limit:
            raise ValueError(f"Non-finite or excessive base norm in block starting {begin}")
        guards = error_guard(qnorm64, max_norm, dim)
        for qs in range(0, nq, query_batch):
            qe = min(qs + query_batch, nq)
            distances = torch.mm(qgpu[qs:qe], bgpu.T)
            distances.mul_(-2).add_(qnorm32[qs:qe, None]).add_(bnorm32[None, :])
            upper = cutoffs[qs:qe].copy()
            if np.isinf(upper).any():
                # The local approximate kth plus E is an upper bound on the
                # exact local kth. Until k rows have been seen, retain all rows.
                if len(bgpu) >= k:
                    kth = torch.topk(distances, k, dim=1, largest=False,
                                     sorted=False).values.amax(dim=1).cpu().numpy()
                    upper = np.minimum(upper, kth.astype(np.float64) + guards[qs:qe])
            limits64 = upper + guards[qs:qe]
            # Round thresholds UP when converting to FP32 for the comparison.
            limits32 = np.nextafter(limits64.astype(np.float32), np.float32(np.inf))
            mask = distances <= torch.from_numpy(limits32).to(device)[:, None]
            count = int(mask.sum().item())
            if not count:
                del distances, mask
                continue
            # Bound nonzero/refinement allocations even if millions of vectors
            # tie, or cancellation makes the FP32 filter uninformative.
            # Enumerating a few extra ID pairs is cheap; don't slice a sparse
            # mask into thousands of narrow tiles just to bound FP64 vectors.
            enumeration_limit = max(refine_batch, 1048576)
            width = (len(bgpu) if count <= enumeration_limit
                     else max(1, enumeration_limit // (qe - qs)))
            for bs in range(0, len(bgpu), width):
                pairs = mask[:, bs:bs + width].nonzero(as_tuple=False)
                for ps in range(0, len(pairs), refine_batch):
                    selected = pairs[ps:ps + refine_batch]
                    qi, bi = selected[:, 0] + qs, selected[:, 1] + bs
                    delta = bgpu.index_select(0, bi).to(torch.float64)
                    delta.sub_(q64gpu.index_select(0, qi))
                    exact = delta.square_().sum(dim=1).cpu().numpy()
                    query_ids = qi.cpu().numpy()
                    base_ids = bi.cpu().numpy() + begin
                    if not np.isfinite(exact).all():
                        raise ValueError("Non-finite FP64 candidate distance")
                    refined += len(exact)
                    # nonzero traverses each query's row contiguously.
                    split = np.flatnonzero(np.diff(query_ids)) + 1
                    for indices in np.split(np.arange(len(exact)), split):
                        if not len(indices):
                            continue
                        q = int(query_ids[indices[0]])
                        best_ids[q], best_distances[q] = merge_topk(
                            best_ids[q], best_distances[q], base_ids[indices], exact[indices], k)
                        if len(best_ids[q]) == k:
                            cutoffs[q] = best_distances[q][-1]
            del distances, mask
        if progress:
            done = begin + len(bgpu)
            elapsed = time.monotonic() - started
            eta = elapsed * (len(base) - done) / done
            print(f"[scan] {done:,}/{len(base):,} base rows; "
                  f"FP64 pairs={refined:,}; elapsed={elapsed:.1f}s; ETA={eta:.1f}s", flush=True)

    ids, distances = np.stack(best_ids), np.stack(best_distances)
    if ids.shape != (nq, k) or not np.isfinite(distances).all():
        raise RuntimeError("Incomplete ground truth")
    return ids, distances, {"scan_seconds": time.monotonic() - started,
                            "fp64_refined_pairs": refined,
                            "exhaustive_pairs": len(base) * nq}


def compare_existing(base, queries, ids, distances, old):
    old_k = old.shape[1]
    if old_k > ids.shape[1]:
        raise ValueError("Existing GT is wider than requested topk")
    report = {"old_topk": old_k, "identical_prefix_queries": 0,
              "identical_topk_set_queries": 0, "overlap_ids": 0,
              "distance_equivalent_queries": 0, "old_distance_inversions": 0,
              "worst_old_distance_inversion": 0.0}
    # Only summation roundoff is treated as equality, not FP32-sized errors.
    for q in range(len(queries)):
        delta = base[old[q]].astype(np.float64) - queries[q].astype(np.float64)
        old_d = np.einsum("ij,ij->i", delta, delta)
        tolerance = 32 * np.finfo(np.float64).eps * base.shape[1] * max(1.0, float(old_d.max()))
        diff = np.diff(old_d)
        inversions = int(np.count_nonzero(diff < -tolerance))
        report["old_distance_inversions"] += inversions
        report["worst_old_distance_inversion"] = min(
            report["worst_old_distance_inversion"], float(diff.min()) if len(diff) else 0.0)
        overlap = len(np.intersect1d(ids[q, :old_k], old[q]))
        report["overlap_ids"] += overlap
        report["identical_topk_set_queries"] += int(overlap == old_k)
        report["identical_prefix_queries"] += int(np.array_equal(ids[q, :old_k], old[q]))
        equivalent = np.all(np.abs(np.sort(old_d) - distances[q, :old_k]) <= tolerance)
        report["distance_equivalent_queries"] += int(equivalent)
    report["mean_id_overlap"] = report["overlap_ids"] / old.size
    report["compatibility"] = "format"
    return report


def publish_new(path: Path, payload_writer):
    """Atomically publish a complete file, refusing to overwrite any file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            payload_writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic no-clobber publication on the same filesystem.
    finally:
        os.unlink(temporary)


def write_ivecs(path: Path, ids: np.ndarray):
    if ids.min() < 0 or ids.max() > np.iinfo(np.int32).max:
        raise ValueError("IDs cannot be represented as signed int32 ivecs")
    records = np.empty((len(ids), ids.shape[1] + 1), dtype="<i4")
    records[:, 0], records[:, 1:] = ids.shape[1], ids
    publish_new(path, lambda stream: records.tofile(stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Float32 fbin; IDs are original row numbers")
    parser.add_argument("--query", required=True, type=Path, help="Float32 fbin, in original query order")
    parser.add_argument("--output", required=True, type=Path, help="New .ivecs file; never overwrites existing files")
    parser.add_argument("--existing-gt", type=Path, help="Old .ivecs for a full top-k compatibility report")
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--device", default="cuda:0", help="cuda:N; cpu is for small correctness tests")
    parser.add_argument("--base-block", type=int, default=262144)
    parser.add_argument("--query-batch", type=int, default=1024)
    parser.add_argument("--refine-batch", type=int, default=32768)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    started = time.monotonic()
    if args.output.suffix.lower() != ".ivecs":
        parser.error("--output must have an .ivecs extension")
    report_path = Path(str(args.output) + ".json")
    if args.output.exists() or report_path.exists():
        parser.error("Output or JSON report already exists; choose a new output path")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    base, queries = read_fbin(args.base), read_fbin(args.query)
    if len(base) - 1 > np.iinfo(np.int32).max:
        parser.error("Base IDs exceed the signed int32 ivecs range")
    old = read_ivecs(args.existing_gt, len(queries), len(base)) if args.existing_gt else None
    if old is not None and old.shape[1] > args.topk:
        parser.error("Existing GT is wider than --topk")
    print(f"[input] base={base.shape}, queries={queries.shape}, squared_l2, topk={args.topk}", flush=True)
    print(f"[compute] device={args.device}; FP32 filter + guarded FP64 refinement; TF32 disabled", flush=True)
    ids, distances, timing = exhaustive_l2(
        base, queries, k=args.topk, device=args.device, base_block=args.base_block,
        query_batch=args.query_batch, refine_batch=args.refine_batch)
    compatibility = None
    if old is not None:
        compatibility = compare_existing(base, queries, ids, distances, old)
        print("[compatibility] " + json.dumps(compatibility, sort_keys=True), flush=True)
    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "metric": "squared_l2", "base_shape": list(base.shape), "query_shape": list(queries.shape),
        "base_bytes": args.base.stat().st_size, "query_bytes": args.query.stat().st_size,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
        "compatibility": compatibility, **timing,
        "total_seconds": time.monotonic() - started,
    }
    write_ivecs(args.output, ids)
    publish_new(report_path, lambda stream: stream.write((json.dumps(metadata, indent=2) + "\n").encode()))
    print(f"[done] {args.output} ({len(ids)} x {args.topk}); report={report_path}", flush=True)
    return metadata


if __name__ == "__main__":
    main()
