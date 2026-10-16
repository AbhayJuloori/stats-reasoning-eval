"""Writes docs/data.json for the results page from the run files and results/summary.json.

Run `python -m evalset.analyze` first; rerun both after new hand grades so the page updates.
"""
import json
import re

from evalset.analyze import FLAWS, load, load_solve
from evalset.llm import slug
from evalset.sandbox import ROOT

RESULTS = ROOT / "results"
JUDGE = "ollama/phi4:14b"


def issue_and_fix(text):
    def grab(label):
        m = re.search(rf"{label}\s*[:\-]\s*\**\s*(.*?)(?:\n\s*\**\s*(?:ISSUE|FIX|VERDICT)\b|\Z)", text or "", re.S | re.I)
        return m.group(1).strip() if m else ""
    return grab("ISSUE") or (text or "").strip()[:600], grab("FIX")


def main():
    summary = json.loads((RESULTS / "summary.json").read_text())
    cases = load(ROOT / "prompts" / "worked_answers.jsonl")
    reviews = load(RESULTS / "reviews.jsonl")
    v2 = {(g["model"], g["review_id"]): g["score"] for g in load(RESULTS / f"judge_v2_grades_{slug(JUDGE)}.jsonl")}
    hand = {(h["model"], h["review_id"]): h["score"] for h in load(RESULTS / "hand_grades.jsonl")}

    by_review = {}
    for r in reviews:
        issue, fix = issue_and_fix(r["response"])
        by_review.setdefault(r["review_id"], []).append({
            "model": r["model"].split("/")[-1], "verdict": r["verdict"], "issue": issue[:700], "fix": fix[:500],
            "score": v2.get((r["model"], r["review_id"])), "hand": hand.get((r["model"], r["review_id"]))})

    site_cases = []
    for c in cases:
        site_cases.append({
            "id": c["case_id"], "flaw": c["flaw"], "task": c["task"], "data": c["data"], "key": c["key"],
            "flawed": {k: c["flawed"][k] for k in ("code", "output", "conclusion")},
            "clean": {k: c["clean"][k] for k in ("code", "output", "conclusion")},
            "reviews": by_review.get(f"{c['case_id']}_flawed", []),
            "control_reviews": by_review.get(f"{c['case_id']}_clean", []),
        })

    skills = {}
    items = {p["id"]: p for p in load(ROOT / "prompts" / "prompts.jsonl")}
    for r in load_solve("prompts"):
        s = skills.setdefault(items[r["id"]]["skill"], [0, 0])
        s[0] += r["correct"]
        s[1] += 1

    out = {
        "flaws": FLAWS,
        "models": sorted(summary["reviews"], key=lambda m: -summary["reviews"][m]["catch_rate"]["rate"]),
        "summary": summary,
        "skills": skills,
        "n_hand": len(hand),
        "cases": site_cases,
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "data.json").write_text(json.dumps(out))
    print(f"docs/data.json: {len(site_cases)} cases, {len(reviews)} reviews, {len(hand)} hand grades")


if __name__ == "__main__":
    main()
