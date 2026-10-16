"""Models answer data questions by writing a Python script that we execute.

Program-of-thought with one repair turn: if the script crashes, the traceback goes back
to the model once. The printed last line is the prediction.

    python -m evalset.run_solve --set prompts --models ollama/qwen3:8b,ollama/gemma3:12b
    python -m evalset.run_solve --set qrdata --sample 100 --models ...
"""
import argparse
import json
import random
from collections import Counter

import pandas as pd

from evalset.grading import item_correct, qrdata_correct
from evalset.llm import Cache, cached_chat, slug
from evalset.sandbox import ROOT, SHEETS, extract_code, last_line, run_code

RESULTS = ROOT / "results"
QRDATA_JSON = ROOT / "data" / "raw" / "QRData" / "benchmark" / "QRData.json"

SYSTEM = ("You are a careful data analyst. You answer questions about data by writing a single "
          "Python script that computes the answer.")

INSTRUCTIONS = ("Write one complete Python script that loads the file(s) by the file name shown "
                "(they are in the current directory), computes the answer with pandas, numpy, scipy, "
                "statsmodels or scikit-learn, and prints ONLY the final answer on the last line. "
                "Return the script in a single ```python code block.")


def preview(filename, rows=5):
    df = pd.read_csv(SHEETS / filename, nrows=2000)
    n_rows = sum(1 for _ in open(SHEETS / filename)) - 1
    dtypes = ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items())
    return (f"File {filename}: {n_rows} rows. Columns: {dtypes}\n"
            f"First {rows} rows:\n{df.head(rows).to_csv(index=False)}")


def build_messages(item):
    previews = "\n".join(preview(f) for f in item["files"])
    user = (f"{item['description']}\n\n{previews}\nQuestion: {item['question']}\n\n{INSTRUCTIONS}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def solve(cache, model, item):
    messages = build_messages(item)
    attempts = []
    for attempt in range(2):
        reply = cached_chat(cache, model, f"{item['id']}#a{attempt}", messages, max_tokens=1200)
        code = extract_code(reply["text"])
        res = run_code(code)
        attempts.append({"code": code, "ok": res["ok"], "stdout": res["stdout"][-1500:],
                         "stderr": res["stderr"][-800:], "seconds": reply.get("seconds"),
                         "completion_tokens": reply.get("completion_tokens")})
        if res["ok"] and last_line(res["stdout"]):
            break
        messages = messages + [
            {"role": "assistant", "content": reply["text"]},
            {"role": "user", "content": "The script failed or printed nothing:\n"
                                        f"{(res['stderr'] or 'no output')[-1200:]}\n"
                                        "Fix it and return the full corrected script in one ```python block."},
        ]
    final = attempts[-1]
    pred = last_line(final["stdout"]) if final["ok"] else ""
    return pred, attempts


def load_set(name, sample, seed):
    if name == "prompts":
        return [json.loads(l) for l in open(ROOT / "prompts" / "prompts.jsonl")]
    raw = json.load(open(QRDATA_JSON))
    items = []
    for idx, q in enumerate(raw):
        items.append({
            "id": f"qrdata_{idx:03d}",
            "files": q["data_files"],
            "description": q["data_description"],
            "question": q["question"],
            "answer": q["answer"],
            "question_type": q["meta_data"]["question_type"],
            "source_ref": q["meta_data"]["reference"],
            "keywords": q["meta_data"]["keywords"],
        })
    if sample and sample < len(items):
        # stratify by source so the sample keeps QRData's mix of statistical and causal items
        rng = random.Random(seed)
        by_source = {}
        for it in items:
            by_source.setdefault(it["source_ref"].split(" ")[0], []).append(it)
        picked = []
        for src, group in sorted(by_source.items()):
            k = round(sample * len(group) / len(items))
            picked += rng.sample(group, min(k, len(group)))
        items = sorted(picked, key=lambda x: x["id"])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["prompts", "qrdata"], required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the output file when several workers run at once")
    args = ap.parse_args()

    items = load_set(args.set, args.sample, args.seed)
    if args.limit:
        items = items[: args.limit]
    out_path = RESULTS / f"solve_{args.set}{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["model"], r["id"]) for r in map(json.loads, out_path.read_text().splitlines())}

    for model in args.models.split(","):
        cache = Cache(RESULTS / "cache" / f"solve_{args.set}_{slug(model)}.jsonl")
        tally = Counter()
        for item in items:
            if (model, item["id"]) in done:
                continue
            pred, attempts = solve(cache, model, item)
            if args.set == "qrdata":
                correct = qrdata_correct(pred, item["answer"], item["question_type"])
            else:
                correct = item_correct(pred, item["answer"], item["tolerance"])
            tally[correct] += 1
            row = {"model": model, "id": item["id"], "pred": pred, "answer": item["answer"],
                   "correct": correct, "n_attempts": len(attempts), "attempts": attempts}
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{model} {item['id']} pred={pred[:30]!r} gold={str(item['answer'])[:20]!r} "
                  f"{'OK' if correct else '--'}", flush=True)
        print(f"== {model}: {tally[True]}/{sum(tally.values())} correct this run", flush=True)


if __name__ == "__main__":
    main()
