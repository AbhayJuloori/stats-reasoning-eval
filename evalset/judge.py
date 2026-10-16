"""LLM judge for model reviews, and a pairwise position bias test.

grade:     scores each review 0/1/2 on rubric/RUBRIC.md given the answer key.
pairwise:  shows the flawed and clean version of the same case side by side, both orders,
           and asks which analysis is sounder. A judge without position bias picks the
           clean one equally often whether it is shown first or second.

    python -m evalset.judge grade --judge ollama/phi4:14b
    python -m evalset.judge grade_v2 --judge ollama/phi4:14b
    python -m evalset.judge pairwise --judge ollama/phi4:14b
"""
import argparse
import json
import re

from evalset.llm import Cache, RateLimited, cached_chat, slug
from evalset.sandbox import ROOT

RESULTS = ROOT / "results"
RUBRIC = (ROOT / "rubric" / "RUBRIC.md").read_text()

GRADE_PROMPT = """You grade a model's review of a data science worked answer using this rubric.

{rubric}

Worked answer type: {kind}
Answer key (what a good reviewer should catch): {key}

The model's review:
<<<
{review}
>>>

Apply the rubric table for this answer type exactly. Respond with a single line of JSON and nothing else:
{{"score": <0, 1 or 2>, "reason": "<one short sentence>"}}"""

CLEAN_KEY = "This worked answer has no planted flaw; the method supports the conclusion."

PAIR_PROMPT = """Two analysts answered the same question. Decide which analysis is methodologically sounder.

=== Analysis A ===
{a}

=== Analysis B ===
{b}

Answer with one letter on the first line: A or B. Then one sentence on why."""

# v2: the verdict is parsed in code; the judge answers one yes/no question per review.
MECH_PROMPT = """A model reviewed a data science worked answer that contains a known methodology flaw.

The known flaw: {key}

The model's review:
<<<
{review}
>>>

Does the review's ISSUE identify this specific flaw, or describe the same underlying mechanism in other words?
A different concern, a generic caution, or the right topic with the wrong mechanism counts as NO.
Answer YES or NO on the first line, then one short sentence."""

FALSE_ALARM_PROMPT = """A model reviewed a data science worked answer whose method is correct and supports its conclusion.

The model's review:
<<<
{review}
>>>

Does the review claim a serious methodological error that would overturn the conclusion? A minor caveat or a
suggestion for extra checks counts as NO. Answer YES or NO on the first line, then one short sentence."""

YES_NO = re.compile(r"\b(YES|NO)\b", re.I)

SCORE = re.compile(r'"score"\s*:\s*([012])')
LETTER = re.compile(r"^\W*(A|B)\b", re.M)


def load(name):
    return [json.loads(l) for l in open(name)]


def grade(judge):
    cases = {c["case_id"]: c for c in load(ROOT / "prompts" / "worked_answers.jsonl")}
    reviews = load(RESULTS / "reviews.jsonl")
    out_path = RESULTS / f"judge_grades_{slug(judge)}.jsonl"
    done = {(r["model"], r["review_id"]) for r in load(out_path)} if out_path.exists() else set()
    cache = Cache(RESULTS / "cache" / f"grade_{slug(judge)}.jsonl")
    for r in reviews:
        if (r["model"], r["review_id"]) in done:
            continue
        flawed = r["planted"] != "none"
        prompt = GRADE_PROMPT.format(
            rubric=RUBRIC,
            kind=f"flawed ({r['planted']})" if flawed else "clean control",
            key=cases[r["case_id"]]["key"] if flawed else CLEAN_KEY,
            review=r["response"].strip()[:3000])
        reply = cached_chat(cache, judge, f"{r['model']}|{r['review_id']}",
                            [{"role": "system", "content": "You are a strict grader. Output one line of JSON only."},
                             {"role": "user", "content": prompt}], max_tokens=300)
        m = SCORE.search(reply["text"])
        row = {"judge": judge, "model": r["model"], "review_id": r["review_id"], "planted": r["planted"],
               "verdict": r["verdict"], "score": int(m.group(1)) if m else None, "raw": reply["text"][-400:]}
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{judge} {r['model']} {r['review_id']} verdict={r['verdict']} score={row['score']}", flush=True)


def grade_v2(judge):
    """Rubric score built from the parsed verdict plus one yes/no judge call (none for SOUND verdicts)."""
    cases = {c["case_id"]: c for c in load(ROOT / "prompts" / "worked_answers.jsonl")}
    reviews = load(RESULTS / "reviews.jsonl")
    out_path = RESULTS / f"judge_v2_grades_{slug(judge)}.jsonl"
    done = {(r["model"], r["review_id"]) for r in load(out_path)} if out_path.exists() else set()
    cache = Cache(RESULTS / "cache" / f"grade_v2_{slug(judge)}.jsonl")
    for r in reviews:
        if (r["model"], r["review_id"]) in done:
            continue
        flawed = r["planted"] != "none"
        answer, raw = None, ""
        if r["verdict"] == "FLAWED":
            prompt = (MECH_PROMPT.format(key=cases[r["case_id"]]["key"], review=r["response"].strip()[:3000])
                      if flawed else FALSE_ALARM_PROMPT.format(review=r["response"].strip()[:3000]))
            reply = cached_chat(cache, judge, f"{r['model']}|{r['review_id']}",
                                [{"role": "user", "content": prompt}], max_tokens=120)
            m = YES_NO.search(reply["text"])
            answer, raw = (m.group(1).upper() if m else None), reply["text"][-300:]
        if flawed:
            score = 0 if r["verdict"] != "FLAWED" else (2 if answer == "YES" else 1)
        else:
            score = 2 if r["verdict"] != "FLAWED" else (0 if answer == "YES" else 1)
        row = {"judge": judge + " (v2)", "model": r["model"], "review_id": r["review_id"],
               "planted": r["planted"], "verdict": r["verdict"], "score": score, "answer": answer, "raw": raw}
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{judge} v2 {r['model']} {r['review_id']} verdict={r['verdict']} score={score}", flush=True)


def pairwise(judge, limit=0):
    cases = load(ROOT / "prompts" / "worked_answers.jsonl")
    if limit:
        # spread a partial run evenly across flaw types
        by_type = {}
        for c in cases:
            by_type.setdefault(c["flaw"], []).append(c)
        per = max(1, limit // len(by_type))
        cases = [c for group in by_type.values() for c in group[:per]]
    out_path = RESULTS / f"pairwise_{slug(judge)}.jsonl"
    done = {(r["case_id"], r["clean_position"]) for r in load(out_path)} if out_path.exists() else set()
    cache = Cache(RESULTS / "cache" / f"pairwise_{slug(judge)}.jsonl")
    for c in cases:
        for clean_position in ["A", "B"]:
            if (c["case_id"], clean_position) in done:
                continue
            a, b = (c["clean"]["text"], c["flawed"]["text"]) if clean_position == "A" else (c["flawed"]["text"], c["clean"]["text"])
            try:
                reply = cached_chat(cache, judge, f"{c['case_id']}|clean_{clean_position}",
                                    [{"role": "user", "content": PAIR_PROMPT.format(a=a, b=b)}], max_tokens=1500)
            except RateLimited as exc:
                print(f"stopping: {exc}", flush=True)
                return
            m = LETTER.search(reply["text"])
            pick = m.group(1) if m else None
            row = {"judge": judge, "case_id": c["case_id"], "flaw": c["flaw"], "clean_position": clean_position,
                   "pick": pick, "picked_clean": pick == clean_position if pick else None,
                   "raw": reply["text"][-400:]}
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{judge} {c['case_id']} clean={clean_position} pick={pick}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["grade", "grade_v2", "pairwise"])
    ap.add_argument("--judge", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "grade":
        grade(args.judge)
    elif args.mode == "grade_v2":
        grade_v2(args.judge)
    else:
        pairwise(args.judge, args.limit)


if __name__ == "__main__":
    main()
