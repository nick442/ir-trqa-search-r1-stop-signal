#!/usr/bin/env python3
"""Report for the 2x2 stop-notice experiment. CPU only, login node is fine.

    python tools/notice_report.py <exp_dir> [<exp_dir> ...] --baseline <exp_dir> [--cut 4] [--out reports]

Give the four cells of one dataset and split, e.g. search_r1_bm25, search_r1_bm25_notice,
search_r1_bm25_novel_passage_d30 and search_r1_bm25_novel_passage_d30_nofill_notice. Recall, searches, EM/SM and
the paired tests use the code of tools/novelty_report.py and the repo's significance_test.py.

Definitions
  ending            how a question ended: a numeric answer; an <answer> without a number (the TRQA metric
                    scores it as wrong); no <answer> because all MAX_ITER turns were used; no <answer> although
                    the model stopped earlier; OOM
  used-up search    a search whose result had nothing new left, by the rule of that run's search tool:
                      baseline (+ notice): none of the shown passages is new in this question (or no hits)
                      memory (+ notice):   fewer than 3 new passages remained in the top D (or no hits)
                    In the notice runs these are exactly the searches that got the notice (read from the log);
                    in the silent runs the same condition is computed from the logs, so the agent's next
                    action can be compared with and without the notice.
  next action       what the agent did right after a used-up search: the same query again, another query it
                    had used before, a new query, an answer (with / without a number), stopped without an
                    answer, or nothing because the turn limit was reached
  cut at K searches the run as if the agent had been stopped after K searches (K=4 is Search-R1's training
                    budget): share of relevant entities shown by then, and EM counting only the questions
                    that had finished within K searches
"""
import argparse
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import novelty_report as nr  # noqa: E402  (same measures, same code)

NUM = re.compile(r'[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\d*\.\d+)')  # as metrics/generation_eval_metrics.py
ENDINGS = ("numeric answer", "answer without a number", "no answer: turn limit", "no answer: stopped early", "oom")
ACTIONS = ("same query again", "earlier query again", "new query", "answer with a number", "answer without a number",
           "stopped without an answer", "turn limit reached", "turn without a search")
PAIRED = ["has_answer", "numeric_answer", "searches", "at_limit", "shown_entity_recall", "strict_at_cut",
          "exact_match", "soft_match_50.0", "soft_match_90.0"]


def used_up(step, seen, topk, memory):
    """Was this search's result used up (see the module docstring)? `seen` = passages shown before it."""
    nt = step.get("notice")
    if isinstance(nt, dict):
        return bool(nt.get("given"))
    ids = [str(x) for x in (step.get("doc_ids") or [])]
    nv = step.get("novelty")
    if isinstance(nv, dict):
        return nv.get("n_new_available", topk) < topk
    if memory:      # memory-only run, search without hits (no novelty log)
        return not ids
    return not any(i not in seen for i in ids)


def analyse(rec, topk, max_iter, memory):
    """Ending of the question and the agent's action after every used-up search."""
    path = [s for s in (rec.get("reasoning_path") or []) if isinstance(s, dict)]
    pred = rec.get("prediction")
    numeric = pred is not None and bool(NUM.search(str(pred)))
    if any(s.get("error") for s in path):
        return {"ending": "oom", "has_answer": 0.0, "numeric_answer": 0.0, "events": [], "at_limit": 0.0}
    turns = [s for s in path if "prediction" not in s]
    if numeric:
        ending = "numeric answer"
    elif pred is not None:
        ending = "answer without a number"
    else:
        ending = "no answer: turn limit" if len(turns) >= max_iter else "no answer: stopped early"
    events, seen, queries, k = [], set(), [], 0
    for i, s in enumerate(turns):
        q = s.get("search_query")
        if not q:
            continue
        k += 1
        if used_up(s, seen, topk, memory):
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            if nxt is None:
                action = ("turn limit reached" if len(turns) >= max_iter else
                          "answer with a number" if numeric else
                          "answer without a number" if pred is not None else "stopped without an answer")
            elif not nxt.get("search_query"):
                action = "turn without a search"
            else:
                nq = nr.norm_query(nxt["search_query"])
                action = ("same query again" if nq == nr.norm_query(q) else
                          "earlier query again" if nq in queries else "new query")
            events.append({"search": k, "action": action})
        queries.append(nr.norm_query(q))
        seen.update(str(x) for x in (s.get("doc_ids") or []))
    return {"ending": ending, "has_answer": float(pred is not None), "numeric_answer": float(numeric),
            "events": events, "at_limit": float(len(turns) >= max_iter), "n_searches": k}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dirs", nargs="+")
    ap.add_argument("--baseline", required=True, help="the silent baseline of the same dataset and split (one of exp_dirs or not)")
    ap.add_argument("--trqa", default=nr.DEFAULT_TRQA)
    ap.add_argument("--out", default=None, help="folder for CSV files (default: none written)")
    ap.add_argument("--cut", type=int, default=4, help="searches of the training budget (default 4)")
    ap.add_argument("--max_iter", type=int, default=20)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    trqa = os.path.abspath(args.trqa)
    metrics = nr._load_module(os.path.join(trqa, "c5_task_evaluation", "metrics", "retrieval_eval_metrics.py"), "trqa_retrieval_metrics")
    entity_of = metrics.extract_entity_id
    sig = nr._load_module(os.path.join(trqa, "c5_task_evaluation", "significance_test.py"), "trqa_significance_test")

    dirs = [os.path.normpath(d) for d in args.exp_dirs]
    base_dir = os.path.normpath(args.baseline)
    if base_dir not in dirs:
        dirs.insert(0, base_dir)
    max_steps = max(args.max_iter, args.cut)
    runs, key0 = [], None
    for d in dirs:
        run, ds, split, exp = nr.parse_exp_dir(d)
        if key0 is None:
            key0 = (ds, split)
            qrels = nr.load_qrels(nr.qrels_path(trqa, ds, split))
        elif (ds, split) != key0:
            sys.exit(f"ERROR: {d} is {ds}/{split}, the others are {key0[0]}/{key0[1]}; give one dataset and split per call")
        records, _ = nr.load_records(d)
        memory = "_novel_" in exp
        rows = []
        for q, r in records.items():
            row = nr.analyse_question(r, qrels.get(q, set()), entity_of, max_steps)
            a = analyse(r, args.topk, args.max_iter, memory)
            row.update({k: a[k] for k in ("ending", "has_answer", "numeric_answer", "at_limit")})
            row["_events"] = a["events"]
            row["strict_at_cut"] = row["_strict_at"][args.cut - 1]
            row["within_cut"] = float(row["searches"] <= args.cut and a["has_answer"] > 0)
            row["em_within_cut"] = row["exact_match"] * row["within_cut"]
            row["used_up_within_cut"] = float(any(e["search"] <= args.cut for e in a["events"]))
            rows.append(row)
        runs.append({"dir": d, "exp": exp, "rows": rows, "notice": exp.endswith("_notice") or "_notice_" in exp, "memory": memory})

    pct = lambda xs: f"{100.0 * sum(xs) / len(xs):.1f}" if xs else "-"
    avg = lambda xs: f"{sum(xs) / len(xs):.2f}" if xs else "-"
    ds, split = key0
    print(f"# Stop-notice report: {ds} {split}\n")
    for r in runs:
        print(f"- `{r['exp']}`: {len(r['rows'])} questions | memory: {'yes' if r['memory'] else 'no'} | notice: {'yes' if r['notice'] else 'no'}")

    print("\n## 1. How questions end, effort, recall and answers\n")
    print("| Experiment | n | " + " | ".join(f"% {e}" for e in ENDINGS) + " | % with an <answer> | avg searches | shown-entity recall (strict) | EM | SM50 | SM90 |")
    print("|---|---|" + "---|" * (len(ENDINGS) + 6))
    for r in runs:
        rows = r["rows"]
        c = Counter(x["ending"] for x in rows)
        print(f"| {r['exp']} | {len(rows)} | " + " | ".join(f"{100.0 * c.get(e, 0) / len(rows):.1f}" for e in ENDINGS)
              + f" | {pct([x['has_answer'] for x in rows])} | {avg([x['searches'] for x in rows])}"
              + f" | {pct([x['shown_entity_recall'] or 0.0 for x in rows])} | {pct([x['exact_match'] for x in rows])}"
              + f" | {pct([x['soft_match_50.0'] for x in rows])} | {pct([x['soft_match_90.0'] for x in rows])} |")

    print("\n## 2. Used-up searches and what the agent does next\n")
    print("In the notice runs a used-up search is a search that got the notice; in the silent runs it is the same "
          "situation without the notice.\n")
    print("| Experiment | % searches used up | % questions with one | first one at search (mean) | searches after the first one (mean) |")
    print("|---|---|---|---|---|")
    for r in runs:
        rows = r["rows"]
        n_s = sum(x["searches"] for x in rows)
        withev = [x for x in rows if x["_events"]]
        print(f"| {r['exp']} | {100.0 * sum(len(x['_events']) for x in rows) / n_s:.1f}" if n_s else f"| {r['exp']} | -", end="")
        print(f" | {pct([float(bool(x['_events'])) for x in rows])} | {avg([x['_events'][0]['search'] for x in withev])}"
              f" | {avg([x['searches'] - x['_events'][0]['search'] for x in withev])} |")
    for title, pick in (("Next action after the FIRST used-up search of a question (% of questions that had one)", lambda ev: ev[:1]),
                        ("Next action after EVERY used-up search (% of used-up searches)", lambda ev: ev)):
        print(f"\n{title}\n")
        print("| Experiment | n | " + " | ".join(ACTIONS) + " |")
        print("|---|---|" + "---|" * len(ACTIONS))
        for r in runs:
            acts = [e["action"] for x in r["rows"] for e in pick(x["_events"])]
            c = Counter(acts)
            print(f"| {r['exp']} | {len(acts)} | " + " | ".join(f"{100.0 * c.get(a, 0) / len(acts):.1f}" if acts else "-" for a in ACTIONS) + " |")

    print(f"\n## 3. Cut at {args.cut} searches (Search-R1's training budget)\n")
    print(f"| Experiment | shown-entity recall after {args.cut} searches | at the end | % of the final recall reached by search {args.cut} "
          f"| % questions answered within {args.cut} searches | EM counting only those | EM of the full run | % questions with a used-up search within {args.cut} |")
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        rows = r["rows"]
        a = sum(x["strict_at_cut"] or 0.0 for x in rows); b = sum(x["shown_entity_recall"] or 0.0 for x in rows)
        print(f"| {r['exp']} | {100.0 * a / len(rows):.2f} | {100.0 * b / len(rows):.2f} | {100.0 * a / b:.0f}" if b else f"| {r['exp']} | - | - | -", end="")
        print(f" | {pct([x['within_cut'] for x in rows])} | {pct([x['em_within_cut'] for x in rows])} | {pct([x['exact_match'] for x in rows])}"
              f" | {pct([x['used_up_within_cut'] for x in rows])} |")

    print("\n## 4. Paired tests (Wilcoxon signed-rank + bootstrap 95% CI of the difference; repo significance_test.py)\n")
    by_exp = {r["exp"]: r for r in runs}
    base = next(r for r in runs if r["dir"] == base_dir)
    pairs = [(base, r) for r in runs if r is not base]
    for r in runs:  # the notice's own effect within the memory row: memory + notice vs memory only
        ref = by_exp.get(r["exp"].replace("_nofill_notice", ""))
        if r["exp"].endswith("_nofill_notice") and ref is not None:
            pairs.append((ref, r))
    for m in PAIRED:
        for r in runs:
            for x in r["rows"]:
                x.setdefault(m, None)
    print("| Comparison (new vs reference) | measure | reference | new | diff | 95% CI | p |")
    print("|---|---|---|---|---|---|---|")
    scale = lambda m: 1.0 if m == "searches" else 100.0
    csv_rows = []
    for ref, r in pairs:
        res, n_diff = nr.paired_tests(sig, ref["rows"], r["rows"], args.bootstrap, metrics=PAIRED)
        for t in res:
            s = scale(t["metric"])
            ci = "-" if t["ci_low"] is None else f"[{s * t['ci_low']:+.2f}, {s * t['ci_high']:+.2f}]"
            p = "-" if t["p"] is None else ("<0.0001" if t["p"] < 0.0001 else f"{t['p']:.4f}")
            print(f"| {r['exp']} vs {ref['exp']} | {t['metric']} | {s * t['mean_base']:.2f} | {s * t['mean_new']:.2f} | {s * t['diff']:+.2f} | {ci} | {p} |")
            csv_rows.append(dict(t, new=r["exp"], reference=ref["exp"]))
        if n_diff:
            print(f"| {r['exp']} vs {ref['exp']} | NOTE: {n_diff} questions are not in both runs and were left out | | | | | |")
    print("\nMeasures in %, except searches. has_answer = the run ended with an <answer>; numeric_answer = that answer holds a "
          f"number; at_limit = all {args.max_iter} turns used; strict_at_cut = shown-entity recall after {args.cut} searches. "
          "No correction for multiple tests.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        stem = os.path.join(args.out, f"notice_{ds}_{split}")
        nr.write_csv(stem + "_paired.csv", csv_rows, ["new", "reference", "metric", "n", "mean_base", "mean_new", "diff", "ci_low", "ci_high", "p"])
        fields = ["exp", "qid", "ending", "has_answer", "numeric_answer", "at_limit", "searches", "shown_entity_recall", "strict_at_cut",
                  "within_cut", "exact_match", "soft_match_50.0", "soft_match_90.0", "n_used_up", "first_used_up_search", "first_next_action"]
        per_q = [dict({k: x.get(k) for k in fields}, exp=r["exp"], n_used_up=len(x["_events"]),
                      first_used_up_search=x["_events"][0]["search"] if x["_events"] else None,
                      first_next_action=x["_events"][0]["action"] if x["_events"] else None) for r in runs for x in r["rows"]]
        nr.write_csv(stem + "_per_question.csv", per_q, fields)
        print(f"\nCSV files: {stem}_paired.csv, {stem}_per_question.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
