"""Exercise index serialization, fbin/ibin IO, and search without external data."""
import argparse
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import tempfile


def write_bin(path, rows, code):
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", len(rows), len(rows[0])))
        for row in rows:
            stream.write(struct.pack("<" + code * len(row), *row))


def write_vecs(path, rows, code):
    with path.open("wb") as stream:
        for row in rows:
            stream.write(struct.pack("<I", len(row)))
            stream.write(struct.pack("<" + code * len(row), *row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexer", required=True)
    parser.add_argument("--search", required=True)
    args = parser.parse_args()
    rng = random.Random(42)
    dim, count, nlist = 32, 1024, 4
    centers = [[rng.uniform(-4, 4) for _ in range(dim)] for _ in range(nlist)]
    base = [[centers[i % nlist][d] + rng.gauss(0, 0.1) for d in range(dim)]
            for i in range(count)]
    selected = [3, 17, 205, 511, 1001]
    queries = [base[i] for i in selected]
    with tempfile.TemporaryDirectory(prefix="coraanns-smoke-") as name:
        root = Path(name)
        write_bin(root / "base.fbin", base, "f")
        write_bin(root / "queries.fbin", queries, "f")
        write_bin(root / "gt.ibin", [[i] for i in selected], "I")
        write_vecs(root / "centers.fvecs", centers, "f")
        write_vecs(root / "assignments.ivecs", [[i % nlist] for i in range(count)], "I")
        env = dict(os.environ, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE")
        subprocess.run([args.indexer, str(root / "base.fbin"),
                        str(root / "centers.fvecs"), str(root / "assignments.ivecs"),
                        "8", str(root / "ivf.index"), "l2", "false"],
                       check=True, env=env)
        result = subprocess.run([args.search, str(root / "ivf.index"),
                                 str(root / "queries.fbin"), str(root / "gt.ibin"),
                                 "--topk", "1", "--nprobes", "4", "--threads", "2",
                                 "--warmup", "0", "--rounds", "1",
                                 "--recall-base", str(root / "base.fbin")],
                                check=True, text=True, stdout=subprocess.PIPE, env=env)
        print(result.stdout)
        rows = re.findall(r"^4\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$", result.stdout, re.M)
        if not rows or float(rows[-1][0]) <= 0 or float(rows[-1][1]) != 1.0:
            raise RuntimeError("CPU round trip did not recover every query's nearest vector")


if __name__ == "__main__":
    main()
