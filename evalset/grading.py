"""Answer matching for numeric and categorical answers.

QRData items use the benchmark's own rule (prediction within 3% of the gold number, or a
prefix match for multiple choice). The written prompt set uses a per item tolerance:
    {"kind": "rel", "rel": 0.03, "decimals": 2}  within 3% or half a rounding unit
    {"kind": "pvalue"}                            both < 0.001, or within 10% / 0.002
    {"kind": "exact"}                             integer or string match
"""
import re

NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE]-?\d+)?%?")


def first_number(text):
    m = NUMBER.search(text or "")
    if not m:
        return None
    s = m.group().replace(",", "")
    if s.endswith("%"):
        return float(s[:-1]) / 100
    return float(s)


def last_number(text):
    found = NUMBER.findall(text or "")
    if not found:
        return None
    s = found[-1].replace(",", "")
    if s.endswith("%"):
        return float(s[:-1])  # written prompts ask for plain numbers; keep the value as printed
    return float(s)


def qrdata_correct(pred, gold, question_type):
    """Same rule as QRData's eval.py, with the bare except narrowed."""
    pred, gold = (pred or "").lower().strip(), gold.lower().strip()
    if question_type == "numerical":
        gold_f = float(gold[:-1]) / 100 if gold.endswith("%") else float(gold)
        pred_f = first_number(pred)
        if pred_f is None:
            return False
        lo, hi = sorted([gold_f * 0.97, gold_f * 1.03])
        return lo < pred_f < hi
    return pred[: len(gold)] == gold


def item_correct(pred_text, gold, tol):
    kind = tol["kind"]
    if kind == "exact" and isinstance(gold, str):
        return gold.lower().strip() in (pred_text or "").lower()
    pred = last_number(pred_text)
    if pred is None:
        return False
    if kind == "exact":
        return round(pred) == round(float(gold))
    gold = float(gold)
    if kind == "pvalue":
        if gold < 0.001:
            return 0 <= pred < 0.001
        return abs(pred - gold) <= max(0.1 * gold, 0.002)
    if kind == "rel":
        half_unit = 0.5 * 10 ** (-tol.get("decimals", 2))
        return abs(pred - gold) <= max(tol.get("rel", 0.03) * abs(gold), half_unit + 1e-12)
    raise ValueError(kind)
