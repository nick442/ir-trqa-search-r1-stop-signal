#!/usr/bin/env python3
"""Print what the agent searched and what the search tool showed, step by step. CPU only.

    python tools/novelty_trace.py <exp_dir> [-n 3] [--qid QID ...] [--think]

Works for any Search-R1 run. For every search: the query (REPEAT = same query as an earlier step of the
question, after lowercasing and collapsing whitespace) and the shown passages. A passage that was already
shown earlier in the question is marked with '*'. Runs of the novelty search tool also show each passage's
rank in the original BM25 list and whether it was new or a fill-up, e.g. `18859-0054@7 new`.
With qrels available, passages that are relevant for the question are marked with '+'.
Runs with the stop notice (search_r1_notice, search_r1_novel_notice) show NOTICE after a search where the
tool added "No more new results for this query." and (no notice) otherwise.
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QRELS = {"wiki1": ("Wiki1", "wiki1"), "ecommerce": ("E-Commerce", "ecommerce")}
SPLITS = {"test": ("test", "test"), "val": ("validation", "val")}


def load_records(exp_dir):
    merged = os.path.join(exp_dir, "merged", "evaluation_results_per_query_metrics.jsonl")
    paths = [merged] if os.path.isfile(merged) else sorted(
        glob.glob(os.path.join(exp_dir, "shard_*", "evaluation_results_per_query_metrics.jsonl")))
    if not paths:
        sys.exit(f"ERROR: no per-query results under {exp_dir}")
    records = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records.setdefault(r["qid"], r)
    return records


def load_gold(exp_dir, trqa):
    m = re.fullmatch(r"(wiki1|ecommerce)_(test|val)", os.path.basename(os.path.dirname(os.path.normpath(exp_dir))))
    if not m:
        return {}
    folder, stem = QRELS[m.group(1)]
    sdir, sshort = SPLITS[m.group(2)]
    path = os.path.join(trqa, "dataset", "TRQA", folder, sdir, f"trec_qrels_{stem}_{sshort}.txt")
    gold = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 4 and int(p[3]) > 0:
                    gold.setdefault(p[0], set()).add(p[2])
    return gold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("-n", type=int, default=3, help="number of questions to print (default 3)")
    ap.add_argument("--qid", nargs="+", help="print these questions instead of the first -n")
    ap.add_argument("--think", action="store_true", help="also print the agent's <think> text per step")
    ap.add_argument("--trqa", default=os.path.join(HERE, os.pardir, "total-recall-qa"))
    args = ap.parse_args()

    records = load_records(args.exp_dir)
    gold = load_gold(args.exp_dir, args.trqa)
    qids = args.qid or list(records)[:args.n]
    for qid in qids:
        r = records.get(qid)
        if r is None:
            print(f"\n== {qid}: not in {args.exp_dir}")
            continue
        g = gold.get(qid, set())
        print(f"\n== {qid} | gold {r.get('gt_answer')} | prediction {r.get('prediction')!r} | EM {r.get('exact_match')}"
              + (f" | {len({p.rsplit('-', 1)[0] for p in g})} relevant entities" if g else ""))
        print(f"   {r.get('query')}")
        seen_q, seen_p, k = set(), set(), 0
        for s in r.get("reasoning_path") or []:
            if not isinstance(s, dict):
                continue
            if s.get("error"):
                print(f"   ERROR: {s['error']}")
                continue
            if args.think and s.get("think"):
                print(f"      think: {' '.join(str(s['think']).split())[:300]}")
            q = s.get("search_query")
            if not q:
                if "prediction" not in s:
                    print("   --  (step without a search)")
                continue
            k += 1
            nq = " ".join(str(q).lower().split())
            ids = [str(x) for x in (s.get("doc_ids") or [])]
            info = {x["id"]: x for x in (s.get("novelty") or {}).get("shown", [])}
            cells = []
            for pid in ids:
                cell = pid + ("*" if pid in seen_p else "") + ("+" if pid in g else "")
                if pid in info:
                    cell += f"@{info[pid]['bm25_rank']} {info[pid]['kind']}"
                cells.append(cell)
            nv = s.get("novelty")
            extra = f"  [{nv['n_new_available']} new of top {nv['n_candidates']}]" if isinstance(nv, dict) else ""
            nt = s.get("notice")
            if isinstance(nt, dict):  # 2x2 experiment: was the stop notice given after these passages?
                extra += "  NOTICE" if nt.get("given") else "  (no notice)"
            print(f"   {k:2d} {'REPEAT' if nq in seen_q else '      '} {' '.join(str(q).split())[:110]!r}")
            print(f"             -> {', '.join(cells) if cells else ('(nothing shown)' if isinstance(nv, dict) else '(no hits)')}{extra}")
            seen_q.add(nq)
            seen_p.update(ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
