"""Terminal tool for grading model reviews by hand on rubric/RUBRIC.md.

Draws a fixed, stratified sample of reviews (same seed every time), shows the answer key
and the model's review, and records a 0/1/2 score. The judge's score is hidden so the
hand grade is independent of it. Safe to quit and resume.

If results/draft_grades.jsonl exists (draft scores with a one line reason from a second
grader), each draft is shown and Enter accepts it; typing 0/1/2 overrides it. Every row
records the draft and whether it was changed.

    python -m evalset.hand_grade            # grade the sample
    python -m evalset.hand_grade --n 80     # sample size (default 80)
"""
import argparse
import json
import random
import textwrap

from evalset.sandbox import ROOT

RESULTS = ROOT / "results"
OUT = RESULTS / "hand_grades.jsonl"
DRAFTS = RESULTS / "draft_grades.jsonl"


def sample(reviews, n, seed=0):
    rng = random.Random(seed)
    strata = {}
    for r in reviews:
        strata.setdefault((r["planted"], r["verdict"]), []).append(r)
    picked = []
    for key in sorted(strata):
        group = strata[key]
        k = max(1, round(n * len(group) / len(reviews)))
        picked += rng.sample(group, min(k, len(group)))
    rng.shuffle(picked)
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    cases = {c["case_id"]: c for c in map(json.loads, open(ROOT / "prompts" / "worked_answers.jsonl"))}
    reviews = [json.loads(l) for l in open(RESULTS / "reviews.jsonl")]
    todo = sample(reviews, args.n)
    drafts = {}
    if DRAFTS.exists():
        drafts = {(d["model"], d["review_id"]): d for d in map(json.loads, DRAFTS.read_text().splitlines())}
    done = set()
    if OUT.exists():
        done = {(r["model"], r["review_id"]) for r in map(json.loads, OUT.read_text().splitlines())}

    print("Rubric: rubric/RUBRIC.md.  Scores: 2 caught / correct, 1 partial / minor, 0 missed / false alarm.")
    if drafts:
        print("A draft score and reason is shown under each review: press Enter to agree, or type 0/1/2 to change it.")
    print("Type s to skip, q to quit.\n")
    for i, r in enumerate(todo, start=1):
        if (r["model"], r["review_id"]) in done:
            continue
        case = cases[r["case_id"]]
        flawed = r["planted"] != "none"
        version = case["flawed"] if flawed else case["clean"]
        print("=" * 100)
        print(f"[{i}/{len(todo)}] {r['review_id']}  ({'planted: ' + r['planted'] if flawed else 'clean control'})")
        print(f"\nTASK: {case['task']}")
        print(f"CONCLUSION GIVEN: {textwrap.fill(version['conclusion'], 100)}")
        if flawed:
            print(f"\nANSWER KEY: {textwrap.fill(case['key'], 100)}")
        print("\nMODEL REVIEW:\n" + r["response"].strip()[:2500])
        draft = drafts.get((r["model"], r["review_id"]))
        if draft:
            print(f"\nDRAFT: {draft['draft_score']}  ({draft['draft_reason']})")
        while True:
            s = input("\nscore 0/1/2" + (" [Enter = agree]" if draft else "") + " > ").strip().lower()
            if s == "" and draft:
                s = str(draft["draft_score"])
            if s in {"0", "1", "2", "s", "q"}:
                break
        if s == "q":
            return
        if s == "s":
            continue
        row = {"model": r["model"], "review_id": r["review_id"], "planted": r["planted"], "score": int(s)}
        if draft:
            row.update(draft_score=draft["draft_score"], changed=int(s) != draft["draft_score"])
            note = input("note (optional, Enter to skip) > ").strip() if row["changed"] else ""
        else:
            note = input("note (optional) > ").strip()
        row["note"] = note
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
    print("Done. Run: python -m evalset.analyze")


if __name__ == "__main__":
    main()
