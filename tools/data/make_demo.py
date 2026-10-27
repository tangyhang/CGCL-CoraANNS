#!/usr/bin/env python3
"""Create a deterministic, small L2 dataset with exact ground truth."""
import argparse
from pathlib import Path

import numpy as np
from vector_io import write_fvecs, write_ivecs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    rng = np.random.default_rng(42)
    base = rng.normal(size=(4096, 32)).astype("float32")
    queries = rng.normal(size=(32, 32)).astype("float32")
    distances = ((queries.astype("float64")[:, None, :] - base.astype("float64")) ** 2).sum(axis=2)
    truth = np.argsort(distances, axis=1, kind="stable")[:, :100].astype("int32")
    args.output.mkdir(parents=True)
    write_fvecs(str(args.output / "base.fvecs"), base)
    write_fvecs(str(args.output / "queries.fvecs"), queries)
    write_ivecs(str(args.output / "groundtruth.ivecs"), truth)


if __name__ == "__main__":
    main()
