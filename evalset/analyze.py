"""Turns the raw run files in results/ into results/summary.json and results/tables.md.

Everything reported here is computed from the run files; nothing is typed in by hand.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint

from evalset.llm import slug
from evalset.sandbox import ROOT

RESULTS = ROOT / "results"
FLAWS = ["peeking", "leakage", "simpsons", "wrong_test", "multiple_comparisons"]


def load(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_solve(name):
    """Solve results, merged across worker files (solve_<set>.jsonl, solve_<set>_gpu0.jsonl, ...)."""
    rows = []
    for path in sorted(RESULTS.glob(f"solve_{name}*.jsonl")):
        rows += load(path)
    return rows


def short(model):
    return model.split("/")[-1]


def rate(k, n):
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    lo, hi = proportion_confint(k, n, method="wilson")
    return {"k": int(k), "n": int(n), "rate": round(k / n, 3), "ci95": [round(lo, 3), round(hi, 3)]}


def fmt(r):
    if not r or r["n"] == 0:
        return "-"
    return f"{100 * r['rate']:.0f}% ({r['k']}/{r['n']})"


# --- solving ------------------------------------------------------------------------------

def solve_summary(name):
    rows = load_solve(name)
    items = {}
    if name == "prompts":
        items = {p["id"]: p for p in load(ROOT / "prompts" / "prompts.jsonl")}
    out = {}
    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model]
        entry = {"overall": rate(sum(r["correct"] for r in mine), len(mine)),
                 "ran_without_error": rate(sum(bool(r["pred"]) for r in mine), len(mine))}
        if name == "prompts":
            by = defaultdict(list)
            for r in mine:
                by[items[r["id"]]["source"]].append(r["correct"])
            entry["by_source"] = {k: rate(sum(v), len(v)) for k, v in by.items()}
        out[model] = entry
    return out


def qrdata_breakdown():
    rows = load_solve("qrdata")
    raw = json.load(open(ROOT / "data" / "raw" / "QRData" / "benchmark" / "QRData.json"))
    kind = {f"qrdata_{i:03d}": ("causal" if "Causality" in q["meta_data"]["keywords"] else "statistical")
            for i, q in enumerate(raw)}
    out = {}
    for model in sorted({r["model"] for r in rows}):
        by = defaultdict(list)
        for r in rows:
            if r["model"] == model:
                by[kind[r["id"]]].append(r["correct"])
        out[model] = {k: rate(sum(v), len(v)) for k, v in by.items()}
    return out


# --- reviews --------------------------------------------------------------------------------

def judge_scores(judge, version="v1"):
    name = "judge_grades" if version == "v1" else "judge_v2_grades"
    return {(g["model"], g["review_id"]): g["score"]
            for g in load(RESULTS / f"{name}_{slug(judge)}.jsonl")}


def review_summary(scores):
    reviews = load(RESULTS / "reviews.jsonl")
    out = {}
    for model in sorted({r["model"] for r in reviews}):
        mine = [r for r in reviews if r["model"] == model]
        flawed = [r for r in mine if r["planted"] != "none"]
        clean = [r for r in mine if r["planted"] == "none"]
        caught = [r for r in flawed if scores.get((model, r["review_id"])) == 2]
        entry = {
            "catch_rate": rate(len(caught), len(flawed)),
            "flag_rate": rate(sum(r["verdict"] == "FLAWED" for r in flawed), len(flawed)),
            "false_alarm_rate": rate(sum(r["verdict"] == "FLAWED" for r in clean), len(clean)),
            "by_flaw": {},
        }
        for f in FLAWS:
            fr = [r for r in flawed if r["planted"] == f]
            entry["by_flaw"][f] = {
                "catch": rate(sum(scores.get((model, r["review_id"])) == 2 for r in fr), len(fr)),
                "flag": rate(sum(r["verdict"] == "FLAWED" for r in fr), len(fr)),
            }
        out[model] = entry
    return out


def judge_vs_verdict(scores):
    """Rubric consistency: a FLAWED verdict on a flawed answer can never be a 0, a SOUND one is always 0."""
    reviews = {(r["model"], r["review_id"]): r for r in load(RESULTS / "reviews.jsonl")}
    bad = total = 0
    for key, s in scores.items():
        r = reviews.get(key)
        if r is None or s is None or r["planted"] == "none":
            continue
        total += 1
        if (r["verdict"] == "FLAWED" and s == 0) or (r["verdict"] == "SOUND" and s != 0):
            bad += 1
    return rate(bad, total)


def agreement(scores):
    hand = load(RESULTS / "hand_grades.jsonl")
    pairs = [(h["score"], scores.get((h["model"], h["review_id"]))) for h in hand]
    pairs = [(a, b) for a, b in pairs if b is not None]
    if len(pairs) < 10:
        return {"n": len(pairs), "note": "hand grades not done yet"}
    h, j = np.array(pairs).T
    hb, jb = (h == 2).astype(int), (j == 2).astype(int)
    return {
        "n": len(pairs),
        "raw_agreement_3class": round(float((h == j).mean()), 3),
        "kappa_3class": round(float(cohen_kappa_score(h, j)), 3),
        "weighted_kappa_3class": round(float(cohen_kappa_score(h, j, weights="linear")), 3),
        "raw_agreement_caught": round(float((hb == jb).mean()), 3),
        "kappa_caught": round(float(cohen_kappa_score(hb, jb)), 3),
    }


# --- position bias --------------------------------------------------------------------------

def pairwise_summary():
    out = {}
    for path in sorted(RESULTS.glob("pairwise_*.jsonl")):
        rows = [r for r in load(path) if r["pick"] in ("A", "B")]
        if not rows:
            continue
        judge = rows[0]["judge"]
        first = [r for r in rows if r["clean_position"] == "A"]
        second = [r for r in rows if r["clean_position"] == "B"]
        by_case = defaultdict(dict)
        for r in rows:
            by_case[r["case_id"]][r["clean_position"]] = r["picked_clean"]
        both = [v for v in by_case.values() if len(v) == 2]
        acc_first = rate(sum(r["picked_clean"] for r in first), len(first))
        acc_second = rate(sum(r["picked_clean"] for r in second), len(second))
        out[judge] = {
            "accuracy": rate(sum(r["picked_clean"] for r in rows), len(rows)),
            "picked_A": rate(sum(r["pick"] == "A" for r in rows), len(rows)),
            "acc_clean_shown_first": acc_first,
            "acc_clean_shown_second": acc_second,
            "position_bias": (round(acc_first["rate"] - acc_second["rate"], 3)
                              if acc_first["n"] and acc_second["n"] else None),
            "consistent_across_orders": rate(sum(v["A"] == v["B"] for v in both), len(both)),
            "correct_in_both_orders": rate(sum(v["A"] and v["B"] for v in both), len(both)),
        }
    return out


def main(judge="ollama/phi4:14b"):
    v1, v2 = judge_scores(judge, "v1"), judge_scores(judge, "v2")
    # catch rates use the v2 grades (verdict parsed in code, one yes/no judge call); v1 is kept for comparison
    scores = v2 if len(v2) >= len(v1) else v1
    summary = {
        "solve_prompts": solve_summary("prompts"),
        "solve_qrdata": solve_summary("qrdata"),
        "qrdata_by_type": qrdata_breakdown(),
        "reviews": review_summary(scores),
        "reviews_graded_by": "v2" if scores is v2 else "v1",
        "reviews_v1": review_summary(v1),
        "judge": judge,
        "judge_rubric_violations": {"v1": judge_vs_verdict(v1), "v2": judge_vs_verdict(v2)},
        "judge_vs_hand": {"v1": agreement(v1), "v2": agreement(v2)},
        "pairwise": pairwise_summary(),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Results tables", "", "Generated by `python -m evalset.analyze` from the run files in `results/`.", ""]
    lines += ["## Written prompt set (150 prompts, code executed, graded against reference solutions)", "",
              "| model | overall | Hillstrom prompts | QRData-sheet prompts | script ran |", "|---|---|---|---|---|"]
    for m, e in summary["solve_prompts"].items():
        bs = e.get("by_source", {})
        lines.append(f"| {short(m)} | {fmt(e['overall'])} | {fmt(bs.get('hillstrom'))} | "
                     f"{fmt(bs.get('qrdata_sheets'))} | {fmt(e['ran_without_error'])} |")
    lines += ["", "## QRData benchmark (stratified sample, official 3% tolerance)", "",
              "| model | overall | statistical | causal |", "|---|---|---|---|"]
    for m, e in summary["solve_qrdata"].items():
        bt = summary["qrdata_by_type"].get(m, {})
        lines.append(f"| {short(m)} | {fmt(e['overall'])} | {fmt(bt.get('statistical'))} | {fmt(bt.get('causal'))} |")
    lines += ["", f"## Planted flaw review (graded by {short(judge)}, {summary['reviews_graded_by']})", "",
              "| model | catch rate | flag rate | false alarms on clean | " + " | ".join(FLAWS) + " |",
              "|---|---|---|---|" + "---|" * len(FLAWS)]
    for m, e in summary["reviews"].items():
        per = " | ".join(fmt(e["by_flaw"][f]["catch"]) for f in FLAWS)
        lines.append(f"| {short(m)} | {fmt(e['catch_rate'])} | {fmt(e['flag_rate'])} | "
                     f"{fmt(e['false_alarm_rate'])} | {per} |")
    lines += ["", "Catch rate with the v1 judge (one call, full rubric) for comparison: " +
              ", ".join(f"{short(m)} {fmt(e['catch_rate'])}" for m, e in summary["reviews_v1"].items()), ""]
    lines += ["## Judge quality", "",
              "v1: the judge reads the rubric and returns a 0/1/2 score. "
              "v2: the verdict is parsed in code and the judge answers one yes/no question.", "",
              "| judge | scores contradicting the verdict | n hand graded | raw agreement (0/1/2) | Cohen's kappa | "
              "weighted kappa | raw agreement (caught?) | kappa (caught?) |",
              "|---|---|---|---|---|---|---|---|"]
    for ver in ["v1", "v2"]:
        a = summary["judge_vs_hand"][ver]
        viol = fmt(summary["judge_rubric_violations"][ver])
        if "kappa_3class" in a:
            lines.append(f"| {short(judge)} {ver} | {viol} | {a['n']} | {a['raw_agreement_3class']} | {a['kappa_3class']} | "
                         f"{a['weighted_kappa_3class']} | {a['raw_agreement_caught']} | {a['kappa_caught']} |")
        else:
            lines.append(f"| {short(judge)} {ver} | {viol} | {a['n']} | - | - | - | - | - |")
    lines += ["", "## Position bias (flawed vs clean twin, both orders)", "",
              "| judge | accuracy | picked A | acc. clean first | acc. clean second | bias | same pick both orders |",
              "|---|---|---|---|---|---|---|"]
    for j, e in summary["pairwise"].items():
        lines.append(f"| {short(j)} | {fmt(e['accuracy'])} | {fmt(e['picked_A'])} | {fmt(e['acc_clean_shown_first'])} | "
                     f"{fmt(e['acc_clean_shown_second'])} | {e['position_bias']} | {fmt(e['consistent_across_orders'])} |")
    (RESULTS / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
