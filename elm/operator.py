"""LLM variation operators for the code archive.

LLMOperator calls the Anthropic API (Claude) to mutate one parent genome or
cross two parents into a child. MockOperator does deterministic textual
tweaks so the whole MAP-Elites loop can be exercised offline (no API key,
no spend) before committing to a real run.

The operator's job: given parent source(s) + their measured behavior, emit a
new, valid, *different* `act(obs, state)` policy.
"""

from __future__ import annotations

import os
import re

from elm.sandbox import PRIMITIVES_CARD, compile_policy, CompileError
from elm.archive import Elite

_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


_OPEN_FENCE = re.compile(r"```(?:python)?\s*\n", re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """Pull the `act`-defining Python out of an LLM response.

    Robust to the common failure modes seen in runs:
      - prose, then a complete ```python ... ``` block (normal case),
      - a block that was TRUNCATED before its closing fence (max_tokens hit):
        we recover everything after the opening fence,
      - bare code with no fences at all.
    Helper defs the model writes before `act` (e.g. build_plan) are preserved
    because we return the whole block, not just from `def act` onward.
    """
    # 1) complete fenced blocks — prefer one that defines act.
    blocks = _CODE_FENCE.findall(text)
    for b in blocks:
        if "def act" in b:
            return b.strip()
    # 2) truncated / unclosed fence: take everything after the opening fence.
    m = _OPEN_FENCE.search(text)
    if m:
        candidate = text[m.end():]
        # drop a trailing closing fence if present
        candidate = candidate.split("```", 1)[0]
        if "def act" in candidate:
            return candidate.strip()
    # 3) any complete block, even without `def act` (let the validator reject).
    if blocks:
        return blocks[0].strip()
    # 4) no fences at all — accept raw text if it looks like the policy.
    if "def act" in text:
        return text.strip()
    return None


def _parent_block(e: Elite, render: str | None = None) -> str:
    m = e.measures
    s = (f"# fitness={e.fitness:.1f}  cityPop={e.city_pop}  "
         f"measures=(m0={m[0]:.3f}, m1={m[1]:.3f})\n"
         f"```python\n{e.source.strip()}\n```")
    if render:
        s += f"\n\nResulting city (ascii, '.'=empty R/C/I=zones '='=road/wire '*'=plant):\n```\n{render}\n```"
    return s


SYSTEM_PROMPT = (
    "You are an expert at evolutionary computation and the Micropolis/SimCity "
    "game engine. You write compact, correct Python policies that place tiles "
    "to grow a city.\n"
    "OUTPUT RULES (important):\n"
    "- Reply with EXACTLY ONE ```python code block and NOTHING before or after "
    "it. No preamble, no explanation outside the block.\n"
    "- Put any reasoning as short comments INSIDE the code.\n"
    "- The block must be complete and self-contained: define `def act(obs, "
    "state):` (plus any helpers it needs). Keep it short enough to finish.\n\n"
    + PRIMITIVES_CARD
)


def build_mutate_prompt(parent: Elite, directive: str,
                        render: str | None = None) -> str:
    return (
        "Here is a parent policy and how it performed:\n\n"
        f"{_parent_block(parent, render)}\n\n"
        f"Task: {directive}\n\n"
        "Write a NEW `act(obs, state)` that is a meaningful variation of the "
        "parent — keep what works, change one or a few ideas. It must compile "
        "and follow the contract. Reply with one ```python block only."
    )


def build_crossover_prompt(p1: Elite, p2: Elite, directive: str,
                           renders=(None, None)) -> str:
    return (
        "Here are two parent policies:\n\n"
        f"PARENT A:\n{_parent_block(p1, renders[0])}\n\n"
        f"PARENT B:\n{_parent_block(p2, renders[1])}\n\n"
        f"Task: {directive}\n\n"
        "Write a NEW `act(obs, state)` that recombines the best ideas from "
        "both parents into a single coherent policy. It must compile and "
        "follow the contract. Reply with one ```python block only."
    )


class OperatorError(Exception):
    def __init__(self, message, raw: str | None = None):
        super().__init__(message)
        self.raw = raw  # the raw LLM text / extracted code, if any, for logging


class LLMOperator:
    """Claude as mutation/crossover operator."""

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 8000,
                 temperature: float = 1.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise OperatorError(
                "anthropic SDK not installed. `/usr/bin/python3 -m pip install "
                "anthropic` and set ANTHROPIC_API_KEY."
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise OperatorError("ANTHROPIC_API_KEY not set in environment.")
        # If we're running inside Claude Code, the env carries ANTHROPIC_AUTH_TOKEN
        # (often an EMPTY string) + ANTHROPIC_BASE_URL/CUSTOM_HEADERS that route
        # the SDK through a gateway. The SDK's auth_headers merge BOTH X-Api-Key
        # and Bearer, so an empty auth_token yields an illegal 'Bearer ' header
        # even when api_key is set — and passing auth_token=None just re-reads
        # the empty env value. So we strip those env vars for this process and
        # pin a clean public-API client with the provided api_key.
        for k in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                  "ANTHROPIC_CUSTOM_HEADERS"):
            os.environ.pop(k, None)
        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url="https://api.anthropic.com",
            default_headers={},
        )

    def _call(self, user_prompt: str) -> str:
        # Cache the big static system prompt (primitives card) across calls.
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=[{
                "type": "text", "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts)

    def _produce(self, prompt: str) -> str:
        text = self._call(prompt)
        src = extract_code(text)
        if src is None:
            raise OperatorError("no code block in LLM response", raw=text)
        try:
            compile_policy(src)  # validate before we hand it to the evaluator
        except CompileError as e:
            raise OperatorError(f"LLM produced invalid policy: {e}", raw=src) from e
        return src

    def mutate(self, parent: Elite, directive: str, render=None) -> str:
        return self._produce(build_mutate_prompt(parent, directive, render))

    def crossover(self, p1: Elite, p2: Elite, directive: str, renders=(None, None)) -> str:
        return self._produce(build_crossover_prompt(p1, p2, directive, renders))


class MockOperator:
    """Offline stand-in: deterministic textual tweaks. Produces compilable,
    behaviorally-different children so the MAP-Elites loop can be tested
    without any API call. Not meant to be good — just valid and varied."""

    _ZONES = ["RESIDENTIAL", "COMMERCIAL", "INDUSTRIAL"]

    def __init__(self, rng):
        import threading
        import numpy as _np
        # Independent generator (seeded once from the shared rng) so it doesn't
        # race with the driver's use of `rng` for parent sampling under threads.
        seed = int(rng.integers(1 << 31)) if rng is not None else 0
        self.rng = _np.random.default_rng(seed)
        self._lock = threading.Lock()  # numpy Generator isn't thread-safe

    def _tweak(self, source: str) -> str:
        with self._lock:
            return self._tweak_locked(source)

    def _tweak_locked(self, source: str) -> str:
        r = self.rng.random()
        out = source
        if "Tool.RESIDENTIAL" in source and r < 0.5:
            # Convert the right-hand flank to a different zone type -> mixes
            # the R/C/I shares, moving the child to a new behavior cell.
            target = self._ZONES[int(self.rng.integers(3))]
            # Replace only the *last* RESIDENTIAL occurrence (the right flank).
            head, _, tail = source.rpartition("Tool.RESIDENTIAL")
            out = head + f"Tool.{target}" + tail
        elif "range(-12, 13, 3)" in source:
            step = int(self.rng.integers(2, 5))
            out = source.replace("range(-12, 13, 3)", f"range(-12, 13, {step})")
        elif "range(-12, 13)" in source:
            half = int(self.rng.integers(6, 16))
            out = source.replace("range(-12, 13)", f"range(-{half}, {half + 1})")
        return out

    def mutate(self, parent: Elite, directive: str, render=None) -> str:
        src = self._tweak(parent.source)
        compile_policy(src)  # sanity
        return src

    def crossover(self, p1: Elite, p2: Elite, directive: str, renders=(None, None)) -> str:
        # Trivial: mutate the fitter parent.
        better = p1 if p1.fitness >= p2.fitness else p2
        return self.mutate(better, directive)


def make_operator(kind: str, rng=None, **kwargs):
    if kind == "mock":
        return MockOperator(rng)
    if kind in ("claude", "llm"):
        return LLMOperator(**kwargs)
    raise ValueError(f"unknown operator {kind!r}")
