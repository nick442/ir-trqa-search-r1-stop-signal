#!/usr/bin/env python3
"""Why did the answers change? Counts unanswered questions and how far numeric answers are off. CPU only.

    python tools/answer_diag.py <baseline_exp_dir> <exp_dir> [<exp_dir> ...] [--words]

Per experiment: questions without a final answer (no <answer>, e.g. the agent searched until MAX_ITER, or OOM),
EM and soft match over all questions and over the answered ones, and where the answered predictions lie
relative to the gold value (prediction / gold). Then, against the first folder (the baseline), which
questions lost or gained a soft match at 90% tolerance and whether the lost ones were unanswered.

--words adds one more table at the end (the output above it is unchanged): answers that consist of a number
word only ("three", "Zero"; zero to twenty) are read as digits and scored with the repo's own
soft_exact_match. This is a diagnostic next to the official metric, which scores such answers as wrong;
it shows how much of the error is the answer's format. Needs the total-recall-qa checkout (--trqa).
"""
import glob
import json
import os
import re
import sys

NUM = re.compile(r'[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\d*\.\d+)')  # as metrics/generation_eval_metrics.py


def number(x):
    if x is None:
        return None
    m = NUM.search(str(x))
    return float(m.group(0).replace(',', '')) if m else None


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


def info(r):
    path = r.get("reasoning_path") or []
    pred, gold = number(r.get("prediction")), number(r.get("gt_answer"))
    sm = r.get("soft_matches") or {}
    return {
        "answered": pred is not None,
        "oom": any(isinstance(s, dict) and s.get("error") for s in path),
        "searches": sum(1 for s in path if isinstance(s, dict) and s.get("search_query")),
        "iterations": sum(1 for s in path if isinstance(s, dict) and "prediction" not in s and not s.get("error")),
        "em": float(r.get("exact_match", 0.0)),
        "sm": {t: bool(sm.get(t) or sm.get(float(t))) for t in ("10.0", "50.0", "90.0")},
        "ratio": (pred / gold) if (pred is not None and gold not in (None, 0.0)) else None,
    }


WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}


def word_number(pred):
    """The number if the prediction is a single number word (case and surrounding punctuation ignored), else None."""
    if pred is None or NUM.search(str(pred)):
        return None
    return WORDS.get(str(pred).strip().strip(".!,;:'\"").strip().lower())


def words_table(dirs, trqa):
    import importlib.util
    path = os.path.join(trqa, "c5_task_evaluation", "metrics", "generation_eval_metrics.py")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: --words needs the repo metric code, not found: {path}")
    spec = importlib.util.spec_from_file_location("trqa_generation_eval_metrics", path)
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)
    print("\nDiagnostic, NOT the official metric: answers that are one number word (zero to twenty), read as digits "
          "and scored with the repo's soft_exact_match\n")
    print("| Experiment | n | answers without a number | of those: one number word | word answers: exact if read as digit "
          "| word answers: within 90% if read | EM official | EM with words read | SM90 official | SM90 with words read |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        recs = load(d)
        n = len(recs)
        nonum = [r for r in recs.values() if r.get("prediction") is not None and number(r.get("prediction")) is None]
        em = sum(float(r.get("exact_match", 0.0)) for r in recs.values())
        sm = sum(1 for r in recs.values() if info(r)["sm"]["90.0"])
        w_n = w_em = w_sm = 0
        for r in nonum:
            w = word_number(r.get("prediction"))
            if w is None:
                continue
            w_n += 1
            w_em += bool(gm.soft_exact_match(str(w), r.get("gt_answer"), decimals=3, tolerance_pct=None)["exact_match"])
            w_sm += bool(gm.soft_exact_match(str(w), r.get("gt_answer"), decimals=3, tolerance_pct=90.0)["soft_match"])
        print(f"| {os.path.basename(os.path.normpath(d))} | {n} | {len(nonum)} | {w_n} | {w_em} | {w_sm} "
              f"| {pct(em, n)} | {pct(em + w_em, n)} | {pct(sm, n)} | {pct(sm + w_sm, n)} |")


def pct(a, b):
    return f"{100 * a / b:.1f}" if b else "-"


def main():
    argv = sys.argv[1:]
    words = "--words" in argv
    trqa = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "total-recall-qa")
    if "--trqa" in argv:
        trqa = argv[argv.index("--trqa") + 1]
        del argv[argv.index("--trqa"):argv.index("--trqa") + 2]
    argv = [a for a in argv if a != "--words"]
    if len(argv) < 2 or argv[0] in ("-h", "--help"):
        sys.exit(__doc__)
    exps = [(os.path.basename(os.path.normpath(d)), {q: info(r) for q, r in load(d).items()}) for d in argv]
    print("| Experiment | n | no final answer | of those: OOM | avg searches (answered / unanswered) | EM all | EM answered "
          "| SM50 all | SM50 answered | SM90 all | SM90 answered |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, x in exps:
        v = list(x.values()); ans = [i for i in v if i["answered"]]; un = [i for i in v if not i["answered"]]
        avg = lambda rows: f"{sum(i['searches'] for i in rows) / len(rows):.1f}" if rows else "-"
        print(f"| {name} | {len(v)} | {len(un)} ({pct(len(un), len(v))}%) | {sum(i['oom'] for i in un)} | {avg(ans)} / {avg(un)} "
              f"| {pct(sum(i['em'] for i in v), len(v))} | {pct(sum(i['em'] for i in ans), len(ans))} "
              f"| {pct(sum(i['sm']['50.0'] for i in v), len(v))} | {pct(sum(i['sm']['50.0'] for i in ans), len(ans))} "
              f"| {pct(sum(i['sm']['90.0'] for i in v), len(v))} | {pct(sum(i['sm']['90.0'] for i in ans), len(ans))} |")

    edges = [("< 0.1", lambda t: t < 0.1), ("0.1 - 0.5", lambda t: 0.1 <= t < 0.5), ("0.5 - 0.9", lambda t: 0.5 <= t < 0.9),
             ("0.9 - 1.1", lambda t: 0.9 <= t <= 1.1), ("1.1 - 2", lambda t: 1.1 < t <= 2), ("2 - 10", lambda t: 2 < t <= 10),
             ("> 10", lambda t: t > 10)]
    print("\nAnswered questions: prediction / gold (share of answered, %; 1.0 = exact)\n")
    print("| Experiment | answered | " + " | ".join(lab for lab, _ in edges) + " | negative or gold = 0 |")
    print("|---|---|" + "---|" * (len(edges) + 1))
    for name, x in exps:
        ans = [i for i in x.values() if i["answered"]]
        pos = [i["ratio"] for i in ans if i["ratio"] is not None and i["ratio"] >= 0]
        print(f"| {name} | {len(ans)} | " + " | ".join(pct(sum(1 for t in pos if f(t)), len(ans)) for _, f in edges)
              + f" | {pct(len(ans) - len(pos), len(ans))} |")

    base_name, base = exps[0]
    print(f"\nAgainst {base_name} (same questions): soft match at 90% tolerance\n")
    print("| Experiment | common | lost | of those: no final answer now | of those: answered but further off | gained "
          "| of those: baseline had no answer |")
    print("|---|---|---|---|---|---|---|")
    for name, x in exps[1:]:
        common = [q for q in x if q in base]
        lost = [q for q in common if base[q]["sm"]["90.0"] and not x[q]["sm"]["90.0"]]
        gained = [q for q in common if not base[q]["sm"]["90.0"] and x[q]["sm"]["90.0"]]
        print(f"| {name} | {len(common)} | {len(lost)} | {sum(1 for q in lost if not x[q]['answered'])} "
              f"| {sum(1 for q in lost if x[q]['answered'])} | {len(gained)} | {sum(1 for q in gained if not base[q]['answered'])} |")
    if words:
        words_table(argv, trqa)
    return 0


if __name__ == "__main__":
    sys.exit(main())
