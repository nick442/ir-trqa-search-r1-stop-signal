#!/usr/bin/env python3
"""Split a queries .jsonl into N shards (round-robin, deterministic) so several
Slurm jobs can run the same experiment in parallel.

Usage: python tools/make_shards.py <queries.jsonl> <out_dir> <num_shards>
Writes <out_dir>/shard_<i>of<N>.jsonl and prints their paths.
"""
import os
import sys


def main():
    src, out_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(src, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        path = os.path.join(out_dir, f"shard_{i}of{n}.jsonl")
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(lines[i::n])
        os.replace(tmp, path)  # atomic: array tasks may run this concurrently
        print(path)


if __name__ == "__main__":
    main()
