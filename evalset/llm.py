"""Thin chat client for local Ollama models and OpenRouter free models, with a JSONL cache.

Model names carry their backend as a prefix:
    ollama/qwen3:8b
    openrouter/nvidia/nemotron-3-super-120b-a12b:free

Every call is cached by (model, key) so a crashed run picks up where it stopped.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Ollama models with a separate reasoning channel. qwen3 thinking is switched off so the
# graded models answer directly; gpt-oss (the judge) keeps a short reasoning pass.
THINK_SETTINGS = {"qwen3": False, "gpt-oss": "low"}


def slug(model):
    """File-name-safe model name: ollama/phi4:14b -> ollama_phi4-14b."""
    return model.replace("/", "_").replace(":", "-")


class RateLimited(Exception):
    pass


def _think_flag(model):
    for prefix, value in THINK_SETTINGS.items():
        if model.startswith(prefix):
            return value
    return None


def unload_other_ollama_models(keep):
    """Keep one model in memory at a time. Two large models together overrun 16 GB of RAM."""
    try:
        loaded = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models", [])
    except requests.RequestException:
        return
    for m in loaded:
        if m["name"] != keep:
            requests.post(f"{OLLAMA_URL}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)


def _ollama_chat(model, messages, max_tokens, temperature, seed):
    unload_other_ollama_models(model)
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens, "num_ctx": 6144},
        "keep_alive": "10m",
    }
    think = _think_flag(model)
    if think is not None:
        body["think"] = think
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=900)
    r.raise_for_status()
    d = r.json()
    return {
        "text": d["message"]["content"],
        "prompt_tokens": d.get("prompt_eval_count"),
        "completion_tokens": d.get("eval_count"),
        "finish_reason": d.get("done_reason"),
    }


def _openrouter_chat(model, messages, max_tokens, temperature, seed):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY to call OpenRouter models")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "seed": seed}
    for attempt in range(6):
        r = requests.post(OPENROUTER_URL, json=body, timeout=300,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 429:
            if "per-day" in r.text or "free-models-per-day" in r.text:
                raise RateLimited(r.text[:300])
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        d = r.json()
        if "choices" not in d:
            raise RuntimeError(str(d)[:300])
        choice = d["choices"][0]
        usage = d.get("usage", {})
        return {
            "text": choice["message"].get("content") or "",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }
    raise RateLimited("still rate limited after retries")


def chat(model, messages, max_tokens=800, temperature=0.0, seed=0):
    backend, _, name = model.partition("/")
    start = time.time()
    if backend == "ollama":
        out = _ollama_chat(name, messages, max_tokens, temperature, seed)
    elif backend == "openrouter":
        out = _openrouter_chat(name, messages, max_tokens, temperature, seed)
    else:
        raise ValueError(f"unknown backend in model name: {model}")
    out["seconds"] = round(time.time() - start, 2)
    return out


class Cache:
    """Append-only JSONL cache. One file per experiment."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["cache_key"]] = row

    @staticmethod
    def make_key(model, item_key, messages):
        digest = hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:12]
        return f"{model}|{item_key}|{digest}"

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, row):
        row = dict(row, cache_key=key)
        self.rows[key] = row
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row


def cached_chat(cache, model, item_key, messages, **kwargs):
    key = Cache.make_key(model, item_key, messages)
    hit = cache.get(key)
    if hit is not None:
        return hit
    out = chat(model, messages, **kwargs)
    return cache.put(key, {"model": model, "item": item_key, **out})
