#!/usr/bin/env python3
"""Train Faiss IVF centroids and assign vectors using the original recipe."""
import argparse
from pathlib import Path
from time import perf_counter

import faiss
from vector_io import read_fbin, read_fvecs, write_fvecs, write_ivecs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--nlist", type=int, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--metric", choices=["l2", "ip"], default="l2")
    args = parser.parse_args()
    if args.nlist <= 0:
        parser.error("--nlist must be positive")
    if args.centroids.resolve() == args.assignments.resolve():
        parser.error("centroids and assignments must have different output paths")
    for path in [args.centroids, args.assignments]:
        if path.exists():
            parser.error(f"output already exists: {path}")
    loaders = {".fbin": read_fbin, ".fvecs": read_fvecs}
    if args.base.suffix.lower() not in loaders:
        parser.error("--base must be .fbin or .fvecs")
    base = loaders[args.base.suffix.lower()](str(args.base))
    if args.nlist > len(base):
        parser.error("--nlist cannot exceed the number of base vectors")
    metric = faiss.METRIC_L2 if args.metric == "l2" else faiss.METRIC_INNER_PRODUCT
    start = perf_counter()
    index = faiss.index_factory(base.shape[1], f"IVF{args.nlist},Flat", metric)
    index.verbose = True
    index.train(base)
    print(f"IVF training: {perf_counter() - start:.3f} seconds")
    centers = index.quantizer.reconstruct_n(0, index.nlist)
    _, assignments = index.quantizer.search(base, 1)
    for path in [args.centroids, args.assignments]:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_fvecs(str(args.centroids), centers)
    write_ivecs(str(args.assignments), assignments)


if __name__ == "__main__":
    main()
