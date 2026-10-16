"""Builds prompts/prompts.jsonl: every prompt plus the answer its reference code prints."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evalset import hillstrom_prompts, sheet_prompts
from evalset.sandbox import ROOT, last_line, run_code

OUT = ROOT / "prompts" / "prompts.jsonl"


def solve(item):
    res = run_code(item["reference_code"], timeout=300)
    if not res["ok"]:
        return item, None, res["stderr"]
    return item, last_line(res["stdout"]), None


def main():
    items = hillstrom_prompts.build() + sheet_prompts.build()
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    failures = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(solve, items))
    rows = []
    for item, answer, err in results:
        if answer is None:
            failures += 1
            print(f"FAILED {item['id']}: {err.strip().splitlines()[-1] if err.strip() else err}", file=sys.stderr)
            continue
        rows.append(dict(item, answer=answer))
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} prompts written to {OUT.relative_to(ROOT)} ({failures} failed)")


if __name__ == "__main__":
    main()
