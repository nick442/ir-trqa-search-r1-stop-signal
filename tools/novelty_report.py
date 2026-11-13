#!/usr/bin/env python3
"""Novelty / redundancy report for Search-R1 runs (Phase 2, RQ1-RQ3). CPU only, login node is fine.

Works on any generation experiment folder, the Phase 1 baseline included:

    python tools/novelty_report.py <exp_dir> [<exp_dir> ...] [--baseline <exp_dir>] [--out reports]

<exp_dir> is $DATA_ROOT/results/<RUN_NAME>/<dataset>_<split>/<exp>; the script reads
<exp_dir>/merged/evaluation_results_per_query_metrics.jsonl (or the shard files if there is no
merged/ yet). Dataset and split come from the folder name, the qrels from total-recall-qa/dataset.
Nothing is written inside the result folders; CSVs go to --out (default ./reports).

Definitions (see the Phase 2 prompt):
  step k            the k-th search call of a question (loop iterations without <search> are skipped)
  shown passages    the doc_ids given to the agent at that step (normally 3)
  repeated query    a search query equal to an earlier query of the same question after lowercasing
                    and collapsing whitespace (exact repeats are reported too)
  repeated passage  a shown passage that was already shown at an earlier step of the same question
  relevant entity   a document with at least one passage in the qrels; document = passage id up to the
                    last '-' (metrics.extract_entity_id; Ecommerce ids look like alcohol-23-0000)
  shown-entity recall (strict, main measure)
                    share of relevant entities for which a relevant passage (qrels) was shown
  shown-entity recall (loose)
                    share of relevant entities for which any passage of that document was shown
                    (this is the definition the repo's entity_recall uses)
  DRA R@k           the repo's metric: entity recall over the interleaved top-1000 of every sub-query
  OOM questions     lost their reasoning path, so they count as 0 shown recall (as EM counts them wrong);
                    the strict recall without them is printed as well.

Runs of the novelty search tool (search_r1_novel) also log what the filter did at every search
(reasoning_path[i].novelty). For those runs the report adds a "Novelty filter" section:
  fill-up           a shown passage that is not new: either already shown ("fill_shown") or, in document mode,
                    another passage of a document that was already shown / is in the same list ("fill_other")
  new candidates    passages (document mode: documents) among the top D that had not been shown yet
  BM25 rank         rank of a shown passage in the step's original BM25 list (1 = what the baseline shows first)
and checks that the logs obey the filter rules (see check_novelty_logs).
"""
import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRQA = os.path.join(HERE, os.pardir, "total-recall-qa")
SM_KEYS = ["10.0", "20.0", "50.0", "90.0"]
DRA_KS = [3, 10, 100, 1000]
QRELS = {  # dataset -> (folder, file stem), same mapping as jobs/common.sh
    "wiki1": ("Wiki1", "wiki1"),
    "ecommerce": ("E-Commerce", "ecommerce"),
}
SPLITS = {"test": ("test", "test"), "val": ("validation", "val")}


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- loading
def parse_exp_dir(exp_dir):
    exp_dir = os.path.normpath(exp_dir)
    exp = os.path.basename(exp_dir)
    ds_split = os.path.basename(os.path.dirname(exp_dir))
    run = os.path.basename(os.path.dirname(os.path.dirname(exp_dir)))
    m = re.fullmatch(r"(wiki1|ecommerce)_(test|val)", ds_split)
    if not m:
        sys.exit(f"ERROR: {exp_dir}: parent folder must be <wiki1|ecommerce>_<test|val>, got {ds_split!r}")
    return run, m.group(1), m.group(2), exp


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
    return records, ("merged" if paths == [merged] else f"{len(paths)} shard file(s), not merged")


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and int(parts[3]) > 0:
                qrels[parts[0]].add(parts[2])
    return qrels


def qrels_path(trqa, ds, split):
    folder, stem = QRELS[ds]
    sdir, sshort = SPLITS[split]
    return os.path.join(trqa, "dataset", "TRQA", folder, sdir, f"trec_qrels_{stem}_{sshort}.txt")


def queries_path(trqa, ds, split):
    folder, stem = QRELS[ds]
    sdir, sshort = SPLITS[split]
    return os.path.join(trqa, "dataset", "TRQA", folder, sdir, f"queries_{stem}_{sshort}.jsonl")


# ----------------------------------------------------------------------------- per question
def norm_query(q):
    return " ".join(str(q).lower().split())


def analyse_question(rec, gold_passages, entity_of, max_steps):
    path = rec.get("reasoning_path") or []
    steps = [s for s in path if isinstance(s, dict) and s.get("search_query")]
    oom = any(isinstance(s, dict) and s.get("error") == "cuda_oom" for s in path)

    gold_entities = {entity_of(p) for p in gold_passages}
    n_ent = len(gold_entities)

    seen_q_norm, seen_q_exact, seen_p, seen_d = set(), set(), set(), set()
    strict_cov, loose_cov = set(), set()
    per_step = []
    n_shown = n_rep_shown = n_rep_q = n_rep_q_exact = 0
    for s in steps:
        q = s["search_query"]
        rep_q = norm_query(q) in seen_q_norm
        rep_q_exact = q in seen_q_exact
        seen_q_norm.add(norm_query(q))
        seen_q_exact.add(q)
        ids = [str(x) for x in (s.get("doc_ids") or [])]
        rep = 0
        for pid in ids:
            if pid in seen_p:
                rep += 1
            seen_p.add(pid)
            d = entity_of(pid)
            seen_d.add(d)
            if pid in gold_passages:
                strict_cov.add(d)
            if d in gold_entities:
                loose_cov.add(d)
        n_shown += len(ids)
        n_rep_shown += rep
        n_rep_q += rep_q
        n_rep_q_exact += rep_q_exact
        per_step.append({
            "repeated_query": int(rep_q), "shown": len(ids), "repeated_shown": rep,
            "strict": len(strict_cov) / n_ent if n_ent else None,
            "loose": len(loose_cov) / n_ent if n_ent else None,
        })

    def at_step(k, key):  # cumulative value after search k (carried forward after the last search)
        if not n_ent:
            return None
        if k == 0 or not per_step:
            return 0.0
        return per_step[min(k, len(per_step)) - 1][key]

    sm = rec.get("soft_matches") or {}
    row = {
        "qid": rec["qid"],
        "n_rel_entities": n_ent,
        "n_rel_passages": len(gold_passages),
        "oom": int(oom),
        "searches": len(steps),
        "repeated_queries": n_rep_q,
        "repeated_queries_exact": n_rep_q_exact,
        "shown": n_shown,
        "repeated_shown": n_rep_shown,
        "distinct_passages": len(seen_p),
        "distinct_documents": len(seen_d),
        "shown_entity_recall": at_step(len(per_step), "strict"),
        "shown_entity_recall_loose": at_step(len(per_step), "loose"),
        "exact_match": float(rec.get("exact_match", 0.0)),
    }
    for t in SM_KEYS:
        row[f"soft_match_{t}"] = 1.0 if (sm.get(t) or sm.get(float(t))) else 0.0
    for k in DRA_KS:
        v = rec.get(f"entity_recall@{k}")
        row[f"dra_entity_recall@{k}"] = None if v is None else float(v)
    row["_steps"] = per_step
    row["_strict_at"] = [at_step(k, "strict") for k in range(1, max_steps + 1)]
    row["_loose_at"] = [at_step(k, "loose") for k in range(1, max_steps + 1)]
    return row


# ----------------------------------------------------------------------------- novelty filter logs
def novelty_steps(rec):
    """[(search number k, step dict)] for the searches that carry a novelty log."""
    out, k = [], 0
    for s in rec.get("reasoning_path") or []:
        if isinstance(s, dict) and s.get("search_query"):
            k += 1
            if isinstance(s.get("novelty"), dict):
                out.append((k, s))
    return out


def analyse_novelty(rec):
    """Per-question numbers from the novelty logs, or None if the question has none."""
    steps = novelty_steps(rec)
    if not steps:
        return None
    per_step = []
    for k, s in steps:
        nv = s["novelty"]
        shown = nv.get("shown") or []
        new_ranks = [x["bm25_rank"] for x in shown if x.get("kind") == "new"]
        per_step.append({
            "step": k, "shown": len(shown), "new": len(new_ranks),
            "fill_shown": sum(1 for x in shown if x.get("kind") == "fill_shown"),
            "fill_other": sum(1 for x in shown if x.get("kind") == "fill_other"),
            "n_new_available": nv.get("n_new_available"), "new_ranks": new_ranks,
            "unit": nv.get("unit"), "depth": nv.get("depth"),
        })
    ranks = [r for st in per_step for r in st["new_ranks"]]
    return {
        "qid": rec["qid"],
        "filtered_searches": len(per_step),
        "shown": sum(st["shown"] for st in per_step),
        "new": sum(st["new"] for st in per_step),
        "fill_shown": sum(st["fill_shown"] for st in per_step),
        "fill_other": sum(st["fill_other"] for st in per_step),
        "searches_with_fillup": sum(1 for st in per_step if st["fill_shown"] + st["fill_other"]),
        "searches_without_new": sum(1 for st in per_step if st["new"] == 0),
        "mean_new_available": mean([st["n_new_available"] for st in per_step]),
        "mean_bm25_rank_new": mean(ranks),
        "max_bm25_rank_new": max(ranks) if ranks else None,
        "_steps": per_step,
    }


def novelty_summary(steps):
    """Aggregate over a list of per-step dicts (all questions of an experiment, or one search number)."""
    shown = sum(st["shown"] for st in steps)
    ranks = sorted(r for st in steps for r in st["new_ranks"])
    n = len(steps)
    return {
        "searches": n,
        "fillup_share": sum(st["fill_shown"] + st["fill_other"] for st in steps) / shown if shown else None,
        "fill_shown_share": sum(st["fill_shown"] for st in steps) / shown if shown else None,
        "fill_other_share": sum(st["fill_other"] for st in steps) / shown if shown else None,
        "searches_with_fillup_share": sum(1 for st in steps if st["fill_shown"] + st["fill_other"]) / n if n else None,
        "searches_without_new_share": sum(1 for st in steps if st["new"] == 0) / n if n else None,
        "mean_new_available": mean([st["n_new_available"] for st in steps]),
        "mean_bm25_rank_new": mean(ranks),
        "median_bm25_rank_new": float(np.median(ranks)) if ranks else None,
        "new_beyond_top3_share": sum(1 for r in ranks if r > 3) / len(ranks) if ranks else None,
    }


def check_novelty_logs(records, entity_of, topk, exp_name):
    """Re-derive the search tool's memory from the logged shown passages and check every logged decision
    against the filter rules. Returns (number of searches checked, list of problems)."""
    problems, n = [], 0
    m = re.search(r"_novel_(passage|document)_d(\d+)", exp_name)
    want_unit, want_depth = (m.group(1), int(m.group(2))) if m else (None, None)
    for qid, rec in records.items():
        seen_p, seen_d = set(), set()
        for s in rec.get("reasoning_path") or []:
            if not (isinstance(s, dict) and s.get("search_query")):
                continue
            ids = [str(x) for x in (s.get("doc_ids") or [])]
            nv = s.get("novelty")
            bad = None
            if not isinstance(nv, dict):
                bad = "search with passages but no novelty log" if ids else None
            else:
                n += 1
                shown = nv.get("shown") or []
                unit, depth = nv.get("unit"), nv.get("depth")
                kinds = [x.get("kind") for x in shown]
                ranks = [x.get("bm25_rank") for x in shown]
                new = [x for x in shown if x.get("kind") == "new"]
                new_docs = [entity_of(x["id"]) for x in new]
                rank_of = {"new": 0, "fill_shown": 1, "fill_other": 2}
                if want_unit and (unit != want_unit or depth != want_depth):
                    bad = f"log says {unit} d{depth}, folder name says {want_unit} d{want_depth}"
                elif [x.get("id") for x in shown] != ids:
                    bad = "logged shown ids differ from the step's doc_ids"
                elif nv.get("fill") is False and (len(shown) != min(topk, nv.get("n_new_available", -1)) or len(new) != len(shown)):
                    bad = f"fill-ups are off: {len(shown)} shown ({len(new)} new) but {nv.get('n_new_available')} new candidates were available"
                elif nv.get("fill") is not False and len(shown) != min(topk, nv.get("n_candidates", 0)):
                    bad = f"{len(shown)} passages shown, expected min({topk}, candidates={nv.get('n_candidates')})"
                elif any(r is None or r < 1 or r > min(depth, nv.get("n_candidates", 0)) for r in ranks):
                    bad = f"BM25 rank outside the top {depth} candidates: {ranks}"
                elif len(new) != min(topk, nv.get("n_new_available", -1)):
                    bad = f"{len(new)} new passages shown but {nv.get('n_new_available')} new candidates were available"
                elif [(rank_of.get(k, 9), r) for k, r in zip(kinds, ranks)] != sorted((rank_of.get(k, 9), r) for k, r in zip(kinds, ranks)):
                    bad = "shown passages are not ordered new -> already shown -> other, each in BM25 order"
                elif unit == "passage" and any(x["id"] in seen_p for x in new):
                    bad = "a passage marked new was already shown"
                elif unit == "passage" and "fill_other" in kinds:
                    bad = "fill_other in passage mode"
                elif unit == "document" and (any(d in seen_d for d in new_docs) or len(set(new_docs)) != len(new_docs)):
                    bad = "a passage marked new comes from a document already shown (or twice from one document)"
                elif any(x["id"] not in seen_p for x in shown if x.get("kind") == "fill_shown"):
                    bad = "a fill_shown passage was not shown before"
                elif any(x["id"] in seen_p or entity_of(x["id"]) not in seen_d | set(new_docs)
                         for x in shown if x.get("kind") == "fill_other"):
                    bad = "a fill_other passage is not an unseen passage of a seen document"
            if bad:
                problems.append(f"{qid}: {bad}")
            seen_p.update(ids)
            seen_d.update(entity_of(i) for i in ids)
    return n, problems


def check_notice_logs(records, topk, exp_name):
    """Stop notice (2x2 experiment): re-derive from the shown passages whether the notice had to be given and
    compare with the log (reasoning_path[i].notice). Without memory: given iff none of the shown passages is
    new in this question. With memory (folder name has _novel_): given iff fewer than topk passages were shown,
    and nothing shown may be a repeat. Returns (searches checked, notices given, problems)."""
    problems, n, given = [], 0, 0
    memory = "_novel_" in exp_name
    for qid, rec in records.items():
        seen_p = set()
        for s in rec.get("reasoning_path") or []:
            if not (isinstance(s, dict) and s.get("search_query")):
                continue
            ids = [str(x) for x in (s.get("doc_ids") or [])]
            nt = s.get("notice")
            n_new = len({i for i in ids if i not in seen_p})
            bad = None
            if not isinstance(nt, dict):
                bad = "search without a notice log"
            else:
                n += 1
                given += bool(nt.get("given"))
                want = (len(ids) < topk) if memory else (n_new == 0)
                if bool(nt.get("given")) != want:
                    bad = f"notice given={nt.get('given')} but {len(ids)} shown, {n_new} new"
                elif nt.get("n_shown") != len(ids) or nt.get("n_new_shown") != n_new:
                    bad = f"log says {nt.get('n_shown')} shown / {nt.get('n_new_shown')} new, the step has {len(ids)} / {n_new}"
                elif memory and n_new != len(ids):
                    bad = "memory + notice showed a passage that was already shown"
            if bad:
                problems.append(f"{qid}: {bad}")
            seen_p.update(ids)
    return n, given, problems


# ----------------------------------------------------------------------------- aggregation
def mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else None


def pct(x, d=2):
    return "-" if x is None else f"{100 * x:.{d}f}"


def num(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def summary(rows):
    shown = sum(r["shown"] for r in rows)
    searches = sum(r["searches"] for r in rows)
    no_oom = [r for r in rows if not r["oom"]]
    s = {
        "n": len(rows),
        "n_oom": sum(r["oom"] for r in rows),
        "avg_searches": mean([r["searches"] for r in rows]),
        "share_max_searches": None,
        "repeated_query_share": sum(r["repeated_queries"] for r in rows) / searches if searches else None,
        "repeated_query_share_exact": sum(r["repeated_queries_exact"] for r in rows) / searches if searches else None,
        "questions_with_repeat": mean([float(r["repeated_queries"] > 0) for r in rows]),
        "repeated_shown_share": sum(r["repeated_shown"] for r in rows) / shown if shown else None,
        "distinct_passages": mean([r["distinct_passages"] for r in rows]),
        "distinct_documents": mean([r["distinct_documents"] for r in rows]),
        "shown_entity_recall": mean([r["shown_entity_recall"] for r in rows]),
        "shown_entity_recall_loose": mean([r["shown_entity_recall_loose"] for r in rows]),
        "shown_entity_recall_no_oom": mean([r["shown_entity_recall"] for r in no_oom]),
        "exact_match": mean([r["exact_match"] for r in rows]),
    }
    for t in SM_KEYS:
        s[f"soft_match_{t}"] = mean([r[f"soft_match_{t}"] for r in rows])
    for k in DRA_KS:
        s[f"dra_entity_recall@{k}"] = mean([r[f"dra_entity_recall@{k}"] for r in rows])
    return s


def per_step_table(rows, max_steps):
    out = []
    for k in range(1, max_steps + 1):
        active = [r["_steps"][k - 1] for r in rows if len(r["_steps"]) >= k]
        shown = sum(a["shown"] for a in active)
        out.append({
            "step": k,
            "n_active": len(active),
            "repeated_query_share": mean([a["repeated_query"] for a in active]),
            "repeated_shown_share": sum(a["repeated_shown"] for a in active) / shown if shown else None,
            "shown_entity_recall": mean([r["_strict_at"][k - 1] for r in rows]),
            "shown_entity_recall_loose": mean([r["_loose_at"][k - 1] for r in rows]),
        })
    return out


def tercile_edges(qrels, entity_of):
    counts = [len({entity_of(p) for p in ps}) for ps in qrels.values()]
    q1, q2 = np.quantile(counts, [1 / 3, 2 / 3])
    return float(q1), float(q2)


def tercile_of(n, edges):
    q1, q2 = edges
    return 0 if n <= q1 else (1 if n <= q2 else 2)


def tercile_labels(edges, max_n):
    q1, q2 = edges
    f = lambda x: str(int(x)) if float(x).is_integer() else f"{x:.1f}"
    return [f"T1 (<= {f(q1)})", f"T2 ({f(q1)} < n <= {f(q2)})", f"T3 (> {f(q2)}, max {max_n})"]


# ----------------------------------------------------------------------------- paired tests
PAIRED_METRICS = ["shown_entity_recall", "shown_entity_recall_loose", "dra_entity_recall@1000",
                  "exact_match", "soft_match_10.0", "soft_match_50.0", "distinct_documents", "searches"]


def paired_tests(sig, base_rows, rows, n_boot, metrics=None):
    a = {r["qid"]: r for r in base_rows}
    b = {r["qid"]: r for r in rows}
    common = sorted(set(a) & set(b))
    res = []
    for m in (metrics or PAIRED_METRICS):
        xa = np.array([np.nan if a[q][m] is None else float(a[q][m]) for q in common])
        xb = np.array([np.nan if b[q][m] is None else float(b[q][m]) for q in common])
        xa, xb = np.nan_to_num(xa, nan=0.0), np.nan_to_num(xb, nan=0.0)  # as significance_test.py does
        if np.allclose(xa, xb):
            p = 1.0
        else:
            try:
                _, p = sig.paired_wilcoxon(xa, xb)
            except ValueError:
                p = None
        lo, hi = sig.bootstrap_ci_diff(xa, xb, n_bootstrap=n_boot) if n_boot else (None, None)
        res.append({"metric": m, "n": len(common), "mean_base": float(xa.mean()), "mean_new": float(xb.mean()),
                    "diff": float(xb.mean() - xa.mean()), "p": None if p is None else float(p),
                    "ci_low": lo, "ci_high": hi})
    return res, len(set(a) ^ set(b))


# ----------------------------------------------------------------------------- output
def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dirs", nargs="+", help="experiment folders .../<RUN_NAME>/<dataset>_<split>/<exp>")
    ap.add_argument("--baseline", help="experiment folder to test the others against (paired, per dataset/split)")
    ap.add_argument("--trqa", default=DEFAULT_TRQA, help="total-recall-qa checkout (qrels, metrics/, significance_test.py)")
    ap.add_argument("--out", default="reports", help="folder for the CSV files (default ./reports)")
    ap.add_argument("--max_steps", type=int, default=20, help="MAX_ITER of the runs (default 20)")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap samples for the CI (0 = off)")
    ap.add_argument("--topk", type=int, default=3, help="passages shown per search, DRA_TOPK (default 3; only used to check novelty logs)")
    args = ap.parse_args()

    trqa = os.path.abspath(args.trqa)
    metrics = _load_module(os.path.join(trqa, "c5_task_evaluation", "metrics", "retrieval_eval_metrics.py"), "trqa_retrieval_metrics")
    entity_of = metrics.extract_entity_id
    sig = _load_module(os.path.join(trqa, "c5_task_evaluation", "significance_test.py"), "trqa_significance_test")

    dirs = list(dict.fromkeys(args.exp_dirs + ([args.baseline] if args.baseline else [])))
    exps = []
    qrels_cache = {}
    for d in dirs:
        run, ds, split, exp = parse_exp_dir(d)
        if (ds, split) not in qrels_cache:
            qrels_cache[(ds, split)] = load_qrels(qrels_path(trqa, ds, split))
        qrels = qrels_cache[(ds, split)]
        records, source = load_records(d)
        n_expected = None
        qp = queries_path(trqa, ds, split)
        if os.path.isfile(qp):
            with open(qp) as f:
                n_expected = sum(1 for l in f if l.strip())
        missing_qrels = [q for q in records if q not in qrels]
        rows = [analyse_question(r, qrels.get(q, set()), entity_of, args.max_steps) for q, r in records.items()]
        for k in DRA_KS:  # OOM questions have no DRA recall; count them as 0, as tools/merge_results.py does
            key = f"dra_entity_recall@{k}"
            if any(r[key] is not None for r in rows):
                for r in rows:
                    if r[key] is None:
                        r[key] = 0.0
        nov_rows = [x for x in (analyse_novelty(r) for r in records.values()) if x]
        nov_check = check_novelty_logs(records, entity_of, args.topk, exp) if (nov_rows or "_novel_" in exp) else None
        has_notice = exp.endswith("_notice") or "_notice_" in exp or any(
            isinstance(st, dict) and "notice" in st for r in records.values() for st in r.get("reasoning_path") or [])
        notice_check = check_notice_logs(records, args.topk, exp) if has_notice else None
        exps.append({"dir": d, "label": f"{run}/{exp}", "run": run, "ds": ds, "split": split, "exp": exp,
                     "rows": rows, "source": source, "n_expected": n_expected, "missing_qrels": missing_qrels,
                     "nov_rows": nov_rows, "nov_check": nov_check, "notice_check": notice_check})

    for e in exps:
        e["sum"] = summary(e["rows"])
        e["sum"]["share_max_searches"] = mean([float(r["searches"] >= args.max_steps) for r in e["rows"]])
        e["steps"] = per_step_table(e["rows"], args.max_steps)

    base = None
    if args.baseline:
        base = next(e for e in exps if os.path.normpath(e["dir"]) == os.path.normpath(args.baseline))

    groups = defaultdict(list)
    for e in exps:
        groups[(e["ds"], e["split"])].append(e)

    for (ds, split), es in sorted(groups.items()):
        qrels = qrels_cache[(ds, split)]
        edges = tercile_edges(qrels, entity_of)
        max_n = max(len({entity_of(p) for p in ps}) for ps in qrels.values())
        print(f"\n## {ds} {split}\n")
        for e in es:
            n = e["sum"]["n"]
            note = "" if e["n_expected"] in (None, n) else f" (INCOMPLETE: {n}/{e['n_expected']})"
            print(f"- `{e['label']}`: {n} questions from {e['source']}{note}")
            if e["missing_qrels"]:
                print(f"  WARNING: {len(e['missing_qrels'])} qids have no qrels, e.g. {e['missing_qrels'][:3]}")
            # Shown passages sit at ranks 1-3 of their step, so they are always inside the repo's interleaved
            # top-1000: loose shown recall can never exceed DRA R@1000. If it does, the logs and qrels disagree.
            bad = [r["qid"] for r in e["rows"] if r["dra_entity_recall@1000"] is not None
                   and r["shown_entity_recall_loose"] is not None
                   and r["shown_entity_recall_loose"] > r["dra_entity_recall@1000"] + 1e-9]
            print(f"  check: loose shown recall <= DRA R@1000 for every question: "
                  + ("OK" if not bad else f"FAILED for {len(bad)}, e.g. {bad[:3]}"))
            if e["nov_check"] is not None:
                n_checked, problems = e["nov_check"]
                print(f"  check: novelty logs obey the filter rules ({n_checked} searches): "
                      + ("OK" if not problems and n_checked else
                         ("FAILED: no novelty logs found" if not problems else f"FAILED for {len(problems)}, e.g. {problems[:3]}")))

            if e["notice_check"] is not None:
                n_checked, n_given, problems = e["notice_check"]
                print(f"  check: stop notices obey the rule ({n_checked} searches, {n_given} with a notice): "
                      + ("OK" if not problems and n_checked else
                         ("FAILED: no notice logs found" if not problems else f"FAILED for {len(problems)}, e.g. {problems[:3]}")))

        print(f"\n### Redundancy and effort (RQ1)\n")
        print("| Experiment | n | OOM | avg searches | % at max searches | % repeated queries (exact) | % questions with a repeat "
              "| % shown passages already shown | distinct passages/q | distinct docs/q |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for e in es:
            s = e["sum"]
            print(f"| {e['label']} | {s['n']} | {s['n_oom']} | {num(s['avg_searches'])} | {pct(s['share_max_searches'], 1)} "
                  f"| {pct(s['repeated_query_share'], 1)} ({pct(s['repeated_query_share_exact'], 1)}) "
                  f"| {pct(s['questions_with_repeat'], 1)} | {pct(s['repeated_shown_share'], 1)} "
                  f"| {num(s['distinct_passages'], 1)} | {num(s['distinct_documents'], 1)} |")

        print(f"\n### Recall and answers (%)\n")
        print("| Experiment | shown-entity recall | (excl. OOM) | loose | DRA R@3 | R@10 | R@100 | R@1000 | EM | SM10 | SM20 | SM50 | SM90 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for e in es:
            s = e["sum"]
            print(f"| {e['label']} | {pct(s['shown_entity_recall'])} | {pct(s['shown_entity_recall_no_oom'])} "
                  f"| {pct(s['shown_entity_recall_loose'])} | "
                  + " | ".join(pct(s[f"dra_entity_recall@{k}"]) for k in DRA_KS) + " | "
                  + pct(s["exact_match"]) + " | " + " | ".join(pct(s[f"soft_match_{t}"]) for t in SM_KEYS) + " |")

        for e in es:
            print(f"\n### Per step: {e['label']}\n")
            print("| step | questions still searching | % repeated queries | % shown already shown | shown-entity recall | loose |")
            print("|---|---|---|---|---|---|")
            for st in e["steps"]:
                print(f"| {st['step']} | {st['n_active']} | {pct(st['repeated_query_share'], 1)} "
                      f"| {pct(st['repeated_shown_share'], 1)} | {pct(st['shown_entity_recall'])} | {pct(st['shown_entity_recall_loose'])} |")

        labels = tercile_labels(edges, max_n)
        print(f"\n### By number of relevant entities (terciles of the {ds} {split} qrels) (RQ3)\n")
        print("| Experiment | tercile | n | EM | SM10 | SM50 | SM90 | shown-entity recall | loose | DRA R@1000 | avg searches |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for e in es:
            for t in range(3):
                rs = [r for r in e["rows"] if tercile_of(r["n_rel_entities"], edges) == t]
                if not rs:
                    continue
                s = summary(rs)
                print(f"| {e['label']} | {labels[t]} | {len(rs)} | {pct(s['exact_match'])} | {pct(s['soft_match_10.0'])} "
                      f"| {pct(s['soft_match_50.0'])} | {pct(s['soft_match_90.0'])} | {pct(s['shown_entity_recall'])} "
                      f"| {pct(s['shown_entity_recall_loose'])} | {pct(s['dra_entity_recall@1000'])} | {num(s['avg_searches'])} |")
            for r in e["rows"]:
                r["tercile"] = tercile_of(r["n_rel_entities"], edges) + 1

        nov_es = [e for e in es if e["nov_rows"]]
        if nov_es:
            print(f"\n### Novelty filter (from the step logs)\n")
            print("| Experiment | unit | D | filtered searches | % shown that are fill-ups (already shown + other) "
                  "| % searches with a fill-up | % searches with no new candidate | new candidates available (mean) "
                  "| BM25 rank of new shown passages: mean | median | % from beyond BM25 top 3 |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for e in nov_es:
                steps = [st for r in e["nov_rows"] for st in r["_steps"]]
                ns = novelty_summary(steps)
                units = sorted({str(st["unit"]) for st in steps}); depths = sorted({str(st["depth"]) for st in steps})
                print(f"| {e['label']} | {'/'.join(units)} | {'/'.join(depths)} | {ns['searches']} "
                      f"| {pct(ns['fillup_share'], 1)} ({pct(ns['fill_shown_share'], 1)} + {pct(ns['fill_other_share'], 1)}) "
                      f"| {pct(ns['searches_with_fillup_share'], 1)} | {pct(ns['searches_without_new_share'], 1)} "
                      f"| {num(ns['mean_new_available'], 1)} | {num(ns['mean_bm25_rank_new'], 1)} "
                      f"| {num(ns['median_bm25_rank_new'], 1)} | {pct(ns['new_beyond_top3_share'], 1)} |")
            for e in nov_es:
                print(f"\n### Novelty filter per step: {e['label']}\n")
                print("| step | filtered searches | % shown that are fill-ups | % searches with no new candidate "
                      "| new candidates available (mean) | BM25 rank of new shown passages (mean) |")
                print("|---|---|---|---|---|---|")
                e["nov_steps"] = []
                for k in range(1, args.max_steps + 1):
                    steps = [st for r in e["nov_rows"] for st in r["_steps"] if st["step"] == k]
                    ns = dict(novelty_summary(steps), step=k)
                    e["nov_steps"].append(ns)
                    print(f"| {k} | {ns['searches']} | {pct(ns['fillup_share'], 1)} | {pct(ns['searches_without_new_share'], 1)} "
                          f"| {num(ns['mean_new_available'], 1)} | {num(ns['mean_bm25_rank_new'], 1)} |")

        if base is not None and (base["ds"], base["split"]) == (ds, split):
            for e in es:
                if e is base:
                    continue
                res, n_unpaired = paired_tests(sig, base["rows"], e["rows"], args.bootstrap)
                print(f"\n### Paired test: {e['label']} vs {base['label']} "
                      f"(Wilcoxon signed-rank + bootstrap 95% CI, as significance_test.py)\n")
                if n_unpaired:
                    print(f"WARNING: {n_unpaired} qids are in only one of the two runs and are left out.\n")
                print("| metric | n | baseline | new | diff | 95% CI | p |")
                print("|---|---|---|---|---|---|---|")
                for x in res:
                    scale = 1 if x["metric"] in ("distinct_documents", "searches") else 100
                    ci = "-" if x["ci_low"] is None else f"[{scale * x['ci_low']:+.2f}, {scale * x['ci_high']:+.2f}]"
                    p = "-" if x["p"] is None else f"{x['p']:.4f}"
                    print(f"| {x['metric']} | {x['n']} | {scale * x['mean_base']:.2f} | {scale * x['mean_new']:.2f} "
                          f"| {scale * x['diff']:+.2f} | {ci} | {p} |")
                write_csv(os.path.join(args.out, f"{ds}_{split}", f"paired_{e['run']}_{e['exp']}_vs_{base['run']}_{base['exp']}.csv"),
                          res, ["metric", "n", "mean_base", "mean_new", "diff", "ci_low", "ci_high", "p"])

        # CSVs
        for e in es:
            prefix = os.path.join(args.out, f"{ds}_{split}", f"{e['run']}_{e['exp']}")
            fields = ["qid", "tercile", "n_rel_entities", "n_rel_passages", "oom", "searches", "repeated_queries",
                      "repeated_queries_exact", "shown", "repeated_shown", "distinct_passages", "distinct_documents",
                      "shown_entity_recall", "shown_entity_recall_loose", "exact_match"] + \
                     [f"soft_match_{t}" for t in SM_KEYS] + [f"dra_entity_recall@{k}" for k in DRA_KS]
            write_csv(prefix + "_per_question.csv", e["rows"], fields)
            write_csv(prefix + "_per_step.csv", e["steps"],
                      ["step", "n_active", "repeated_query_share", "repeated_shown_share",
                       "shown_entity_recall", "shown_entity_recall_loose"])
            long_rows = [{"qid": r["qid"], "step": k, "shown_entity_recall": r["_strict_at"][k - 1],
                          "shown_entity_recall_loose": r["_loose_at"][k - 1]}
                         for r in e["rows"] for k in range(1, args.max_steps + 1)]
            write_csv(prefix + "_recall_per_step_per_question.csv", long_rows,
                      ["qid", "step", "shown_entity_recall", "shown_entity_recall_loose"])
            if e["nov_rows"]:
                write_csv(prefix + "_novelty_per_question.csv", e["nov_rows"],
                          ["qid", "filtered_searches", "shown", "new", "fill_shown", "fill_other", "searches_with_fillup",
                           "searches_without_new", "mean_new_available", "mean_bm25_rank_new", "max_bm25_rank_new"])
                write_csv(prefix + "_novelty_per_step.csv", e["nov_steps"],
                          ["step", "searches", "fillup_share", "fill_shown_share", "fill_other_share",
                           "searches_with_fillup_share", "searches_without_new_share", "mean_new_available",
                           "mean_bm25_rank_new", "median_bm25_rank_new", "new_beyond_top3_share"])

    print(f"\nCSV files written to {os.path.abspath(args.out)}/<dataset>_<split>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
