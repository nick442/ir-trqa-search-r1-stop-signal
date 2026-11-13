#!/usr/bin/env python3
"""Merge the per-shard outputs of one generation experiment into a single folder
and recompute the aggregated metrics (same definitions as run_evalution.py).

Usage: python tools/merge_results.py <experiment_dir> <queries.jsonl>
  <experiment_dir> contains shard_*/evaluation_results_per_query_metrics.jsonl
Writes <experiment_dir>/merged/{evaluation_results_per_query_metrics.jsonl,
                                evaluation_results_metrics.json, evaluation_results.txt}
"""
import glob
import json
import os
import sys

TOLERANCES = ["1.0", "5.0", "10.0", "20.0", "50.0", "90.0"]


def main():
    exp_dir, queries_path = sys.argv[1], sys.argv[2]
    with open(queries_path, encoding="utf-8") as f:
        expected = [json.loads(l)["id"] for l in f if l.strip()]

    records = {}
    for path in sorted(glob.glob(os.path.join(exp_dir, "shard_*", "evaluation_results_per_query_metrics.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records.setdefault(r["qid"], r)  # a resumed shard never repeats a qid, but be safe

    trec = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "shard_*", "evaluation_results.txt"))):
        with open(path) as f:
            trec += [l.split() for l in f if l.strip()]

    missing = [q for q in expected if q not in records]
    n = len(records)
    if n == 0:
        print("no results found under", exp_dir)
        return 1

    def mean(key_fn):
        vals = [key_fn(r) for r in records.values()]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    metrics = {
        "n": n,
        "n_expected": len(expected),
        "complete": not missing,
        "exact_match": mean(lambda r: float(r.get("exact_match", 0.0))),
        "soft_exact_match": {t: mean(lambda r, t=t: float(bool(r.get("soft_matches", {}).get(t, False)))) for t in TOLERANCES},
    }
    recall_keys = sorted({k for r in records.values() for k in r if k.startswith("entity_recall")})
    for k in recall_keys:
        metrics[k] = mean(lambda r, k=k: float(r[k]) if k in r else 0.0)
    metrics["n_cuda_oom"] = sum(1 for r in records.values() if any(st.get("error") == "cuda_oom" for st in r.get("reasoning_path", []) if isinstance(st, dict)))
    novs = [st["novelty"] for r in records.values() for st in r.get("reasoning_path", [])
            if isinstance(st, dict) and isinstance(st.get("novelty"), dict)]
    if novs:  # runs of the novelty search tool (search_r1_novel); full report: tools/novelty_report.py
        shown = [x for nv in novs for x in nv.get("shown", [])]
        new_ranks = [x["bm25_rank"] for x in shown if x.get("kind") == "new"]
        metrics["novelty"] = {
            "unit": sorted({str(nv.get("unit")) for nv in novs}),
            "depth": sorted({nv.get("depth") for nv in novs if nv.get("depth") is not None}),
            "filtered_searches": len(novs),
            "fillup_share_of_shown": round(sum(1 for x in shown if x.get("kind") != "new") / len(shown), 4) if shown else None,
            "searches_with_fillup_share": round(sum(1 for nv in novs if nv.get("n_fillups")) / len(novs), 4),
            "mean_new_candidates_available": round(sum(nv.get("n_new_available", 0) for nv in novs) / len(novs), 3),
            "mean_bm25_rank_of_new_shown": round(sum(new_ranks) / len(new_ranks), 3) if new_ranks else None,
        }
    notices = [st["notice"] for r in records.values() for st in r.get("reasoning_path", [])
               if isinstance(st, dict) and isinstance(st.get("notice"), dict)]
    if notices:  # runs with the stop notice (search_r1_notice, search_r1_novel_notice)
        metrics["notice"] = {
            "searches": len(notices),
            "searches_with_notice_share": round(sum(1 for x in notices if x.get("given")) / len(notices), 4),
            "questions_with_notice_share": round(sum(1 for r in records.values() if any(
                isinstance(st, dict) and (st.get("notice") or {}).get("given") for st in r.get("reasoning_path", []))) / len(records), 4),
        }
    steps = [sum(1 for s in r.get("reasoning_path", []) if s.get("search_query")) for r in records.values()]
    if any(steps):
        metrics["avg_search_calls"] = sum(steps) / len(steps)

    out = os.path.join(exp_dir, "merged")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "evaluation_results_per_query_metrics.jsonl"), "w", encoding="utf-8") as f:
        for q in expected:
            if q in records:
                f.write(json.dumps(records[q]) + "\n")
    with open(os.path.join(out, "evaluation_results_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if trec:
        trec.sort(key=lambda p: (p[0], int(p[3])))
        with open(os.path.join(out, "evaluation_results.txt"), "w") as f:
            for p in trec:
                f.write(" ".join(p) + "\n")

    print(json.dumps(metrics, indent=2))
    if missing:
        print(f"WARNING: {len(missing)} of {len(expected)} queries not finished yet "
              f"(resubmit the same job to resume), e.g. {missing[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
