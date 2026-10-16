"""Each model reviews every worked answer in prompts/review_set.jsonl.

The prompt never says a flaw was planted: a clean control looks the same as a flawed
answer, so false alarms can be measured.

    python -m evalset.run_review --models ollama/qwen3:8b,ollama/gemma3:12b
"""
import argparse
import json
import re

from evalset.llm import Cache, cached_chat, slug
from evalset.sandbox import ROOT

RESULTS = ROOT / "results"

REVIEW_PROMPT = """You are reviewing a junior data scientist's worked answer before it goes to a stakeholder.
Check whether the method supports the conclusion: study design, test choice, data handling,
and how the numbers are interpreted. Do not nitpick style.

{text}

Reply in exactly this format:
VERDICT: SOUND or FLAWED
ISSUE: <one or two sentences naming the main methodological problem, or "none">
FIX: <one or two sentences on how to do it correctly, or "none">"""

VERDICT = re.compile(r"VERDICT\s*[:\-]\s*\**\s*(SOUND|FLAWED)", re.I)


def parse_verdict(text):
    m = VERDICT.search(text or "")
    if m:
        return m.group(1).upper()
    upper = (text or "").upper()
    if "FLAWED" in upper and "SOUND" not in upper:
        return "FLAWED"
    if "SOUND" in upper and "FLAWED" not in upper:
        return "SOUND"
    return "UNPARSED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(ROOT / "prompts" / "review_set.jsonl")]
    if args.limit:
        items = items[: args.limit]
    out_path = RESULTS / "reviews.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["model"], r["review_id"]) for r in map(json.loads, out_path.read_text().splitlines())}

    for model in args.models.split(","):
        cache = Cache(RESULTS / "cache" / f"review_{slug(model)}.jsonl")
        for it in items:
            if (model, it["review_id"]) in done:
                continue
            messages = [{"role": "user", "content": REVIEW_PROMPT.format(text=it["text"])}]
            reply = cached_chat(cache, model, it["review_id"], messages, max_tokens=600)
            row = {"model": model, "review_id": it["review_id"], "case_id": it["case_id"],
                   "planted": it["planted"], "verdict": parse_verdict(reply["text"]),
                   "response": reply["text"], "seconds": reply.get("seconds")}
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{model} {it['review_id']} -> {row['verdict']}", flush=True)


if __name__ == "__main__":
    main()
