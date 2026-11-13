#!/usr/bin/env python3
"""How do questions without a final answer end? CPU only.

    python tools/unanswered_diag.py <exp_dir> [<exp_dir> ...] [--max_iter 20] [--examples 3]

"No final answer" is defined as in tools/answer_diag.py: no number can be read from the prediction.
The Search-R1 loop runs at most MAX_ITER turns. A turn with <search> is a search; a turn that neither
closes a <search> tag nor ends the generation (the model wrote until its token limit) also uses up a turn.
The loop ends early when the model ends its output, with or without an <answer>.

Categories of unanswered questions:
  oom            the question hit a CUDA out-of-memory error
  limit_all      used all MAX_ITER turns, every turn was a search
  limit_mixed    used all MAX_ITER turns, some turns had no search (so fewer than MAX_ITER searches)
  stop_no_tag    the model ended its output before the limit without an <answer> tag
  stop_no_number the model gave an <answer> that holds no number
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

NUM = re.compile(r'[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\d*\.\d+)')  # as tools/answer_diag.py
CATS = ("oom", "limit_all", "limit_mixed", "stop_no_tag", "stop_no_number")


def load(exp_dir):
    merged = os.path.join(exp_dir, "merged", "evaluation_results_per_query_metrics.jsonl")
    paths = [merged] if os.path.isfile(merged) else sorted(
        glob.glob(os.path.join(exp_dir, "shard_*", "evaluation_results_per_query_metrics.jsonl")))
    recs = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs.setdefault(r["qid"], r)
    if not recs:
        sys.exit(f"ERROR: no per-query results under {exp_dir}")
    return recs


def classify(r, max_iter):
    """Returns None for an answered question, else a dict describing how it ended."""
    pred = r.get("prediction")
    if pred is not None and NUM.search(str(pred)):
        return None
    path = [s for s in (r.get("reasoning_path") or []) if isinstance(s, dict)]
    if any(s.get("error") for s in path):
        return {"cat": "oom", "turns": 0, "searches": 0, "nosearch": 0, "last": "", "pred": pred}
    turns = [s for s in path if "prediction" not in s]
    final = next((s for s in path if "prediction" in s), {})
    searches = sum(1 for s in turns if s.get("search_query"))
    nosearch = len(turns) - searches
    if len(turns) >= max_iter:
        cat = "limit_all" if nosearch == 0 else "limit_mixed"
    else:
        cat = "stop_no_tag" if pred is None else "stop_no_number"
    return {"cat": cat, "turns": len(turns), "searches": searches, "nosearch": nosearch,
            "nohits": sum(1 for s in turns if s.get("search_query") and not s.get("doc_ids")),
            "last": " ".join(str(final.get("think") or "").split()), "pred": pred}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dirs", nargs="+")
    ap.add_argument("--max_iter", type=int, default=20)
    ap.add_argument("--examples", type=int, default=3, help="examples printed per category and experiment")
    ap.add_argument("--chars", type=int, default=160, help="characters of the last <think> shown per example")
    args = ap.parse_args()

    rows = []
    for d in args.exp_dirs:
        ds = os.path.basename(os.path.dirname(os.path.normpath(d)))
        name = f"{ds}/{os.path.basename(os.path.normpath(d))}"
        recs = load(d)
        un = {q: c for q, c in ((q, classify(r, args.max_iter)) for q, r in recs.items()) if c}
        rows.append((name, len(recs), un))

    print("| Experiment | n | no final answer | " + " | ".join(CATS) + " |")
    print("|---|---|---|" + "---|" * len(CATS))
    for name, n, un in rows:
        c = Counter(x["cat"] for x in un.values())
        print(f"| {name} | {n} | {len(un)} | " + " | ".join(str(c.get(k, 0)) for k in CATS) + " |")

    print("\nUnanswered questions per category: mean turns / mean searches / mean turns without a search\n")
    print("| Experiment | " + " | ".join(CATS[1:]) + " |")
    print("|---|" + "---|" * (len(CATS) - 1))
    for name, n, un in rows:
        cells = []
        for k in CATS[1:]:
            xs = [x for x in un.values() if x["cat"] == k]
            m = lambda key: sum(x[key] for x in xs) / len(xs)
            cells.append(f"{m('turns'):.1f} / {m('searches'):.1f} / {m('nosearch'):.1f}" if xs else "-")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\nstop_no_number: what the <answer> held (most frequent first)\n")
    for name, n, un in rows:
        c = Counter(" ".join(str(x["pred"]).split())[:60] for x in un.values() if x["cat"] == "stop_no_number")
        if c:
            print(f"- {name}: " + "; ".join(f"{k!r} x{v}" for k, v in c.most_common(8)))

    if args.examples:
        print(f"\nExamples (qid | turns/searches | end of the last <think>, {args.chars} chars)\n")
        for name, n, un in rows:
            by = defaultdict(list)
            for q in sorted(un):
                by[un[q]["cat"]].append(q)
            for k in CATS[1:]:
                for q in by[k][:args.examples]:
                    x = un[q]
                    print(f"- {name} | {k} | {q} | {x['turns']}/{x['searches']} | ...{x['last'][-args.chars:]!r}"
                          + (f" | answer={str(x['pred'])[:60]!r}" if x["pred"] is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
