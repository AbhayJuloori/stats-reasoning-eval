"""Runs every planted-flaw case and its corrected twin, then writes:

prompts/worked_answers.jsonl  one row per case: flawed and clean write ups plus the answer key
prompts/review_set.jsonl      what the models review: 60 flawed answers + 15 clean controls
"""
import json
import re
import sys

import pandas as pd

from evalset import flaw_cases
from evalset.sandbox import ROOT, SHEETS, run_code

OUT_CASES = ROOT / "prompts" / "worked_answers.jsonl"
OUT_REVIEW = ROOT / "prompts" / "review_set.jsonl"
FIELD = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*): (.+)$")
PER_TYPE = 12
CONTROLS_PER_TYPE = 3


def render(task, data, code, output, conclusion):
    return (f"Task: {task}\n\nData: {data}\n\nAnalysis code:\n```python\n{code.strip()}\n```\n\n"
            f"Output:\n```\n{output.strip()}\n```\n\nConclusion: {conclusion}")


def execute(code, conclusion):
    res = run_code(code, timeout=300)
    if not res["ok"]:
        raise RuntimeError(res["stderr"][-600:])
    fields = {}
    for line in res["stdout"].splitlines():
        m = FIELD.match(line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return res["stdout"], conclusion.format(**fields), fields


def main():
    hill = pd.read_csv(SHEETS / "hillstrom.csv")
    groups = {
        "peeking": flaw_cases.peeking_cases(hill, PER_TYPE),
        "leakage": flaw_cases.leakage_cases(),
        "simpsons": flaw_cases.simpsons_cases(),
        "wrong_test": flaw_cases.wrong_test_cases(),
        "multiple_comparisons": flaw_cases.multiple_comparison_cases(hill, PER_TYPE),
    }
    cases, failed = [], 0
    for flaw, items in groups.items():
        if len(items) != PER_TYPE:
            print(f"WARNING {flaw}: {len(items)} cases, expected {PER_TYPE}", file=sys.stderr)
        for n, c in enumerate(items, start=1):
            cid = f"{flaw}_{n:02d}"
            try:
                f_out, f_concl, f_fields = execute(c["flawed_code"], c["flawed_conclusion"])
                c_out, c_concl, c_fields = execute(c["clean_code"], c["clean_conclusion"])
            except Exception as exc:  # report and keep going so one bad case is easy to spot
                failed += 1
                print(f"FAILED {cid}: {exc}", file=sys.stderr)
                continue
            cases.append({
                "case_id": cid, "flaw": flaw, "task": c["task"], "data": c["data"], "key": c["key"],
                "flawed": {"code": c["flawed_code"], "output": f_out, "conclusion": f_concl, "fields": f_fields,
                           "text": render(c["task"], c["data"], c["flawed_code"], f_out, f_concl)},
                "clean": {"code": c["clean_code"], "output": c_out, "conclusion": c_concl, "fields": c_fields,
                          "text": render(c["task"], c["data"], c["clean_code"], c_out, c_concl)},
            })
            print(f"{cid}: flawed {f_fields} | clean {c_fields}")

    OUT_CASES.write_text("".join(json.dumps(c) + "\n" for c in cases))

    review = []
    for c in cases:
        review.append({"review_id": f"{c['case_id']}_flawed", "case_id": c["case_id"],
                       "planted": c["flaw"], "text": c["flawed"]["text"]})
    for flaw in groups:
        own = [c for c in cases if c["flaw"] == flaw]
        for c in own[:: max(1, len(own) // CONTROLS_PER_TYPE)][:CONTROLS_PER_TYPE]:
            review.append({"review_id": f"{c['case_id']}_clean", "case_id": c["case_id"],
                           "planted": "none", "text": c["clean"]["text"]})
    OUT_REVIEW.write_text("".join(json.dumps(r) + "\n" for r in review))
    print(f"{len(cases)} cases, {len(review)} review items, {failed} failed")


if __name__ == "__main__":
    main()
