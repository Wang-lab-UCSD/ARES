#!/usr/bin/env python
"""Shared helpers for the ARES report -> label extractor.

These four pieces used to live inside larger, now-removed files (llm_classify.py,
llm_dichotomy.py, score_pairs.py). The extractor only ever needed these bits of them, so they are
collected here and the rest of those files is dropped.

  _parse_json   strip <think>/markdown fences and pull the JSON object out of an LLM reply
  VALID         the set of valid dichotomy labels
  _llm_d        one chat-completions call with retry, used by the mechanism classifier
  resolve_dir   map a (cell, target, partner) triple to its ARES production report directory
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

VALID = {"SEQUENCE", "PROTEIN", "CONTEXT", "UNRESOLVED", "ARTIFACT"}

# ARES production reports, one directory per resolved dependency. Override with the ARES_REPORTS
# environment variable to point at a copy elsewhere.
PROD = Path(os.environ.get("ARES_REPORTS", "/new-stg/home/hanbei/ARES/outputs_production"))

PROVIDERS = {
    "minimax": dict(base_url="https://api.minimax.io/v1", key_env="MINIMAX_API_KEY",
                    model="MiniMax-M3", max_tokens=40000),
    "deepseek": dict(base_url="https://api.deepseek.com", key_env="DEEPSEEK_API_KEY",
                     model="deepseek-chat", max_tokens=8192),
    "deepseek-think": dict(base_url="https://api.deepseek.com", key_env="DEEPSEEK_API_KEY",
                           model="deepseek-chat", max_tokens=32000,
                           extra_body={"thinking": {"type": "enabled"}}),
}


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from an LLM reply, tolerating <think> blocks and code fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object found in:\n{text[:400]}")
    return json.loads(m.group(0))


def _llm_d(prompt: str, provider: str = "minimax", max_attempts: int = 6) -> dict:
    """One chat-completions call at temperature 0 with exponential-backoff retry."""
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"])
    if not key:
        raise RuntimeError(f"{cfg['key_env']} not set")
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    backoffs = [10, 20, 40, 80, 160]
    last = None
    for attempt in range(max_attempts):
        try:
            extra = {"extra_body": cfg["extra_body"]} if cfg.get("extra_body") else {}
            resp = client.chat.completions.create(
                model=cfg["model"], max_tokens=cfg["max_tokens"], temperature=0.0,
                messages=[{"role": "user", "content": prompt}], **extra,
            )
            return _parse_json(resp.choices[0].message.content or "")
        except Exception as e:
            last = e
            if attempt >= max_attempts - 1:
                break
            time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
    raise RuntimeError(f"{provider} call failed: {last!r}")


_DIRCACHE: dict[str, dict[str, Path]] = {}


def resolve_dir(cell: str, target: str, partner: str) -> Path | None:
    """Directory of the ARES production report for one dependency, or None if not found."""
    if cell not in _DIRCACHE:
        m: dict[str, Path] = {}
        cdir = PROD / cell
        if cdir.is_dir():
            for d in cdir.iterdir():
                if d.is_dir() and (d / "state_final.json").exists():
                    stem = d.name[: -(len(cell) + 1)] if d.name.endswith("_" + cell) else d.name
                    m[stem.upper()] = d
        _DIRCACHE[cell] = m
    return _DIRCACHE[cell].get(f"{target}_{partner}".upper())
