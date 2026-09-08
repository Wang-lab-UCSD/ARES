"""Code generation agent — REPL-style incremental execution (v2)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.llm.base import LLMProvider, Message
from src.prompts.coding import (
    build_repl_system_prompt,
    build_repl_initial_prompt,
    build_inspection_prompt,
    build_coding_system_prompt,
)
from src.utils.logging import get_logger


# MiniMax M2.7 occasionally enters a "protocol-confusion" state where it spams
# a single control-flow token thousands of times instead of closing its turn
# cleanly. Across three observed runs, three different tokens have triggered
# the same underlying failure:
#
#   ELK3   (Apr 14): `<invoke name="..."><parameter ...>...</parameter></invoke>`
#                    — Anthropic tool-call syntax (wrong framework)
#   ZNF445 (Apr 15): `<end_turn>` repeated ~4700 times per response
#                    — MiniMax's own turn-terminator (wrong protocol)
#   PTTG1  (Apr 15): `</execute>` dangling close tags, thousands per response
#                    — orphan close tags with no matching open
#
# Left alone, each junk response grows to ~50-90KB and gets saved into the
# coding agent's conversation history. Within 3-10 REPL turns the cumulative
# history exceeds MiniMax's 204,800-token context window and the run dies
# with TokenLimitExceeded. We strip all three patterns before the response
# is appended to history. None of the three patterns can appear in
# legitimate Python code, thinking, or solution text, so the strip never
# touches valid content.


# `<invoke name="..." ...>...</invoke>` blocks (attributes may be quoted with
# " or ' or absent entirely).
_INVOKE_TAG_RE = re.compile(
    r"<invoke\b[^>]*>.*?</invoke\s*>",
    re.DOTALL | re.IGNORECASE,
)

# `<end_turn>` markers (with or without self-closing slash, any case).
# The trailing `\s*` consumes any whitespace/newlines the model may have
# emitted between consecutive tokens in the runaway pattern — collapsing
# the 4700-tag ZNF445 response from ~52KB to a few hundred bytes.
_END_TURN_RE = re.compile(
    r"<end_turn\s*/?>\s*",
    re.IGNORECASE,
)

# Matches a single `<execute ...>` or `</execute ...>` tag plus any
# trailing whitespace. Used by the orphan-close stripper to walk the
# response and track open/close balance.
_EXECUTE_TAG_RE = re.compile(
    r"<(/?)execute\b[^>]*>\s*",
    re.IGNORECASE,
)

# DeepSeek-V4-Flash internal delimiter, leaking into visible output immediately before its
# `<execute>` or `</execute>` tag (fullwidth vertical bar U+FF5C, doubled on each side of the
# literal "DSML"). Landing inside the block it breaks the block's Python with a SyntaxError;
# landing outside it is inert but still noise. Seen in 11 of the demo run's first ~60 REPL turns.
_DSML_TOKEN_RE = re.compile(
    r"</?\uFF5C\uFF5CDSML\uFF5C\uFF5C>\s*",
    re.IGNORECASE,
)

# The same delimiter fused into a tag rather than standing on its own. Two shapes turn up, and
# they need opposite treatment. `<\uFF5C\uFF5CDSML\uFF5C\uFF5Cexecute>` stands where `<execute>`
# belongs and the block behind it is ordinarily perfect Python, so the tag is repaired and only
# the delimiter dropped -- deleting the match would throw the turn away, which is what made these
# read as tagless responses and pushed the REPL loop toward giving up. Every other filling seen
# (`e`, `execution`, `018`, a hex id) names no tag the parser wants, so those are removed whole
# rather than guessed at.
_DSML_FUSED_TAG_RE = re.compile(
    r"<(/?)\uFF5C\uFF5CDSML\uFF5C\uFF5C\.?(execute|solution)>",
    re.IGNORECASE,
)
_DSML_FUSED_JUNK_RE = re.compile(
    r"</?\uFF5C\uFF5CDSML\uFF5C\uFF5C[^>]*>\s*",
    re.IGNORECASE,
)


def _strip_invoke_tags(content: str) -> tuple[str, int]:
    """Remove `<invoke>...</invoke>` blocks from an LLM response.

    Returns (cleaned_content, n_removed). Legitimate content — thinking,
    execute blocks, solution, plain text — is unchanged.
    """
    if not content or "<invoke" not in content.lower():
        return content, 0
    n_removed = 0

    def _sub(_m: re.Match) -> str:
        nonlocal n_removed
        n_removed += 1
        return ""

    cleaned = _INVOKE_TAG_RE.sub(_sub, content)
    return cleaned, n_removed


def _strip_end_turn_tokens(content: str) -> tuple[str, int]:
    """Remove `<end_turn>` markers from an LLM response.

    `<end_turn>` is MiniMax's internal turn-terminator token. The ARES
    REPL protocol uses `<solution>` to end a turn, so `<end_turn>` is
    never valid content — any occurrence is junk.
    """
    if not content or "<end_turn" not in content.lower():
        return content, 0
    n_removed = 0

    def _sub(_m: re.Match) -> str:
        nonlocal n_removed
        n_removed += 1
        return ""

    cleaned = _END_TURN_RE.sub(_sub, content)
    return cleaned, n_removed


def _strip_orphan_execute_closes(content: str) -> tuple[str, int]:
    """Remove orphan `</execute>` close tags using open/close balance tracking.

    Walks the response left-to-right counting `<execute>` opens and
    `</execute>` closes. A close tag is "orphan" if the balance counter
    is already zero (no unmatched open to pair with). Strips only orphan
    closes; legitimate balanced pairs are preserved byte-for-byte even
    when spam is directly adjacent.

    Observed failure: the Apr 15 PTTG1 crash had one legitimate balanced
    `<execute>...</execute>` block directly followed by ~15,000 orphan
    close tags. A naive "strip runs of 3+ closes" regex would eat the
    legitimate close as the first of the run. The balance-aware approach
    correctly keeps the legitimate close and strips only the orphans.

    Returns (cleaned, n_removed). A minimum threshold of 3 orphans is
    required to strip — a single stray close (very rare but not
    impossible in unusual but valid content) is left in place.
    """
    if not content or "</execute" not in content.lower():
        return content, 0

    balance = 0
    orphan_ranges: list[tuple[int, int]] = []  # (start, end) for each orphan match

    for m in _EXECUTE_TAG_RE.finditer(content):
        is_close = m.group(1) == "/"
        if is_close:
            if balance > 0:
                balance -= 1
            else:
                orphan_ranges.append((m.start(), m.end()))
        else:
            balance += 1

    # Conservative threshold: only strip if 3+ orphans found, avoiding
    # accidental matches on unusual-but-legitimate single stray closes.
    if len(orphan_ranges) < 3:
        return content, 0

    # Build cleaned content by skipping orphan ranges (including their
    # trailing whitespace, which the _EXECUTE_TAG_RE pattern consumes).
    parts: list[str] = []
    last_end = 0
    for start, end in orphan_ranges:
        parts.append(content[last_end:start])
        last_end = end
    parts.append(content[last_end:])
    cleaned = "".join(parts)
    return cleaned, len(orphan_ranges)


def _strip_dsml_tokens(content: str) -> tuple[str, int]:
    """Remove DeepSeek's leaked `<\uFF5C\uFF5CDSML\uFF5C\uFF5C>` delimiter from a response.

    Unlike the MiniMax patterns above this one does not run away -- one or two copies per
    response, not thousands -- so it needs no balance tracking, just removal.
    """
    if not content or "dsml" not in content.lower():
        return content, 0
    n_removed = 0

    def _sub(_m: re.Match) -> str:
        nonlocal n_removed
        n_removed += 1
        return ""

    # Repair first, so a salvageable tag is not swept up by the catch-all below.
    cleaned, n_fused = _DSML_FUSED_TAG_RE.subn(r"<\1\2>", content)
    n_removed += n_fused
    cleaned = _DSML_TOKEN_RE.sub(_sub, cleaned)
    cleaned = _DSML_FUSED_JUNK_RE.sub(_sub, cleaned)
    return cleaned, n_removed


def _strip_runaway_tokens(content: str) -> tuple[str, dict[str, int]]:
    """Apply all four runaway-pattern strippers to a response.

    Returns (cleaned_content, counts) where counts is a dict with keys
    `invoke`, `end_turn`, `orphan_close`, and `dsml` giving the number of
    each pattern removed. Applied in sequence; later strippers see content
    with earlier patterns already removed, which is safe because the
    three patterns don't overlap.
    """
    c1, n_invoke = _strip_invoke_tags(content)
    c2, n_end_turn = _strip_end_turn_tokens(c1)
    c3, n_orphan = _strip_orphan_execute_closes(c2)
    c4, n_dsml = _strip_dsml_tokens(c3)
    return c4, {
        "invoke": n_invoke,
        "end_turn": n_end_turn,
        "orphan_close": n_orphan,
        "dsml": n_dsml,
    }


class CodingAgent:
    """Agent that verifies a hypothesis via an incremental REPL conversation.

    Instead of generating one large script and retrying from scratch on errors,
    the agent runs a conversation loop:

      1. Send hypothesis + context to the LLM.
      2. LLM responds with <think>...</think> (reasoning) and one or more
         <execute>...</execute> blocks (Python to run immediately).
      3. Execute each block in the shared Jupyter kernel.
      4. Append the output as <observation>...</observation>.
      5. Repeat until the LLM emits <solution>...</solution>.

    Variables set in one <execute> block persist for all subsequent blocks because
    the kernel maintains its namespace across calls — just like Biomni's REPL.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.logger = get_logger("coding_agent")
        # Set by orchestrator before each hypothesis so debug filenames are labelled
        self.state_iter: int = 1

    # ------------------------------------------------------------------
    # File inspection (one-shot, used for file familiarisation)
    # ------------------------------------------------------------------

    async def generate_inspection_code(self, data_manifest: dict[str, Any]) -> str:
        """Generate a one-shot script that inspects manifest files and prints summaries."""
        self.logger.info("Generating file inspection code")
        prompt = build_inspection_prompt(data_manifest)
        system = build_coding_system_prompt([])
        response = await self.llm.complete([
            Message.system(system),
            Message.user(prompt),
        ])
        code = self._extract_code(response.content)
        self.logger.info("Generated inspection code", {"code_length": len(code)})
        return code

    # ------------------------------------------------------------------
    # REPL verification loop
    # ------------------------------------------------------------------

    async def run_repl(
        self,
        hypothesis: dict[str, Any],
        executor: Any,
        data_manifest: dict[str, Any],
        run_dir: Path,
        prior_evidence: list[dict[str, Any]] | None = None,
        file_summaries: dict[str, str] | None = None,
        allowed_packages: list[str] | None = None,
        max_iterations: int = 20,
    ) -> dict[str, Any]:
        """Verify a hypothesis via an incremental REPL conversation.

        Returns a result dict with: execution_success, support_level, confidence,
        finding, reasoning, findings, issues, summary, _raw_output.
        """
        debug_dir = run_dir / "debug_code"
        debug_dir.mkdir(exist_ok=True)

        # Use the hypothesis's required_data list to select which helper bundles
        # to load. If the hypothesis didn't declare required_data, the dispatcher
        # falls back to loading all bundles (safe default).
        required_data = hypothesis.get("required_data") or None
        system_prompt = build_repl_system_prompt(required_data=required_data)
        self.logger.info("Built REPL system prompt", {
            "required_data_count": len(required_data) if required_data else 0,
            "system_prompt_chars": len(system_prompt),
        })
        initial_msg = build_repl_initial_prompt(
            hypothesis=hypothesis,
            data_manifest=data_manifest,
            prior_evidence=prior_evidence,
            file_summaries=file_summaries,
            allowed_packages=allowed_packages,
        )

        messages: list[Message] = [
            Message.system(system_prompt),
            Message.user(initial_msg),
        ]

        all_stdout_parts: list[str] = []
        no_tag_count = 0

        iteration = 0       # counts only actual code executions
        llm_turns = 0      # counts all LLM calls (including narration-only)
        _MAX_LLM_TURNS = max_iterations * 2  # hard cap to prevent infinite loops
        while iteration < max_iterations and llm_turns < _MAX_LLM_TURNS:
            llm_turns += 1
            self.logger.info("REPL iteration", {
                "iteration": iteration + 1,
                "llm_turn": llm_turns,
                "hypothesis": hypothesis.get("name", "N/A"),
            })

            response = await self.llm.complete(messages)
            content = response.content

            # Save the RAW LLM turn (pre-cleaning) so post-mortem debugging can
            # still inspect whatever the model actually emitted, including any
            # hallucinated junk.
            try:
                # turn, not iteration: `iteration` advances only on a turn that executed
                # something, so naming by it lets a tagless response be overwritten by the next
                # turn -- which is exactly the response worth keeping when diagnosing why the
                # loop gave up.
                (debug_dir / f"iter{self.state_iter}_repl{iteration}_turn{llm_turns}_llm.txt"
                 ).write_text(content, encoding="utf-8")
            except Exception:
                pass

            # Strip MiniMax runaway-token junk BEFORE any downstream processing.
            # Three observed failure patterns, all the same underlying bug —
            # the model spams a single control token thousands of times:
            #   <invoke>...</invoke>    (Anthropic tool-call syntax, ELK3 Apr 14)
            #   <end_turn>              (MiniMax terminator, ZNF445 Apr 15)
            #   </execute></execute>... (orphan close spam, PTTG1 Apr 15)
            # Left alone, each response balloons to 50-90KB and fills the
            # conversation history within 3-10 REPL turns, triggering
            # TokenLimitExceeded. None of these patterns can appear in
            # legitimate Python code, thinking, or solution text.
            content, strip_counts = _strip_runaway_tokens(content)
            if any(v > 0 for v in strip_counts.values()):
                self.logger.warning(
                    "REPL: stripped runaway tokens from response",
                    {
                        "iteration": iteration,
                        "invoke_tags_removed": strip_counts["invoke"],
                        "end_turn_tokens_removed": strip_counts["end_turn"],
                        "orphan_close_tags_removed": strip_counts["orphan_close"],
                        "dsml_tokens_removed": strip_counts["dsml"],
                        "cleaned_length": len(content),
                    },
                )

            # Extract execute blocks first — if present, run them even if a
            # <solution> also appears in the same message (prevents hallucination
            # where the model writes code + solution without actually executing).
            # Also accept <execute code>, <execute python>, etc.
            execute_blocks = re.findall(
                r"<execute[^>]*>(.*?)</execute>", content, re.DOTALL | re.IGNORECASE
            )

            # Solution → done (only when no execute blocks to run)
            if not execute_blocks:
                solution_match = re.search(
                    r"<solution>(.*?)</solution>", content, re.DOTALL | re.IGNORECASE
                )
                if solution_match:
                    self.logger.info("REPL: solution received", {"iteration": iteration})
                    return self._parse_solution(solution_match.group(1).strip(), all_stdout_parts)

            if not execute_blocks:
                no_tag_count += 1
                self.logger.warning("REPL: no tags in response", {"iteration": iteration})
                messages.append(Message.assistant(content))
                if no_tag_count >= 3:
                    # Model is stuck — force a conclusion
                    break
                messages.append(Message.user(
                    "Please either run code with <execute>...</execute> "
                    "or give your conclusion with <solution>...</solution>."
                ))
                continue

            no_tag_count = 0
            iteration += 1  # count only actual code executions toward the budget

            # Execute only the FIRST block per LLM turn — true incremental REPL.
            # The model must see each result before writing the next step.
            # Multiple blocks in one response would skip intermediate feedback,
            # causing cascade failures when an early block fails silently.
            code = execute_blocks[0].strip()
            block_idx = 0

            # Strip unused execute blocks from the stored conversation history.
            # Some models (e.g. MiniMax) emit dozens of execute blocks per response
            # — only the first is run, and storing all of them blows up context size.
            stored_content = content
            if len(execute_blocks) > 1:
                # Keep only the first <execute>...</execute> block
                first_exec_pattern = re.compile(
                    r"<execute[^>]*>.*?</execute>", re.DOTALL | re.IGNORECASE
                )
                first_match = first_exec_pattern.search(content)
                if first_match:
                    # Replace all execute blocks with just the first one
                    parts: list[str] = []
                    last_end = 0
                    for i, m in enumerate(first_exec_pattern.finditer(content)):
                        if i == 0:
                            parts.append(content[last_end:m.end()])
                            last_end = m.end()
                        else:
                            parts.append(content[last_end:m.start()])
                            last_end = m.end()
                    parts.append(content[last_end:])
                    stored_content = "".join(parts)
                    self.logger.warning("REPL: stripped extra execute blocks from history", {
                        "iteration": iteration,
                        "blocks_total": len(execute_blocks),
                        "original_chars": len(content),
                        "stored_chars": len(stored_content),
                    })

            messages.append(Message.assistant(stored_content))
            try:
                (debug_dir / f"iter{self.state_iter}_repl{iteration}_block{block_idx}.py"
                 ).write_text(code, encoding="utf-8")
            except Exception:
                pass

            self.logger.debug("REPL: executing block", {
                "iteration": iteration, "block": block_idx,
                "preview": code[:200],
            })

            result = await executor.execute(code)
            output = result.get_display_output()
            all_stdout_parts.append(output)
            obs = output.strip() if output.strip() else (
                "(no output)" if result.success else f"Error: {result.error}"
            )

            self.logger.info("REPL: block done", {
                "iteration": iteration, "block": block_idx,
                "success": result.success,
            })

            messages.append(Message.user(f"<observation>\n{obs}\n</observation>"))

        # Max iterations — ask for forced conclusion
        self.logger.warning("REPL: requesting forced conclusion", {
            "hypothesis": hypothesis.get("name", "N/A"),
        })
        messages.append(Message.user(
            "You have used all available iterations. "
            "Based on what you have observed so far, give your final conclusion:\n\n"
            "<solution>\n"
            "support_level: SUPPORTS|REJECTS|INCONCLUSIVE\n"
            "confidence: 0.0-1.0\n"
            "finding: <one sentence>\n"
            "reasoning: <explanation>\n"
            "</solution>"
        ))
        response = await self.llm.complete(messages)
        sol = re.search(r"<solution>(.*?)</solution>", response.content, re.DOTALL | re.IGNORECASE)
        if sol:
            return self._parse_solution(sol.group(1).strip(), all_stdout_parts)

        return {
            "execution_success": False,
            "findings": [],
            "support_level": "INCONCLUSIVE",
            "confidence": 0.0,
            "reasoning": "Max REPL iterations reached without a conclusion.",
            "issues": ["Max iterations reached"],
            "summary": "Inconclusive: max iterations reached",
            "_raw_output": "\n".join(all_stdout_parts),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_solution(self, text: str, stdout_parts: list[str]) -> dict[str, Any]:
        def _field(key: str, default: str = "") -> str:
            m = re.search(rf"^{key}\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
            return m.group(1).strip() if m else default

        raw = _field("support_level", "INCONCLUSIVE").upper()
        valid = {"SUPPORTS", "REJECTS", "INCONCLUSIVE", "ERROR", "UNTESTABLE"}
        level = raw if raw in valid else "INCONCLUSIVE"

        try:
            conf = float(_field("confidence", "0.5"))
            conf = max(0.0, min(1.0, conf))
        except ValueError:
            conf = 0.5

        analysis = _field("analysis", "")
        finding = _field("finding", "")
        reasoning = _field("reasoning", text)
        raw_output = "\n".join(stdout_parts)

        return {
            "execution_success": level not in ("ERROR",),
            "analysis": analysis,
            "findings": [finding] if finding else [],
            "support_level": level,
            "confidence": conf,
            "reasoning": reasoning,
            "issues": [],
            "summary": finding or reasoning[:200],
            "_raw_output": raw_output,
        }

    @staticmethod
    def _extract_code(response: str) -> str:
        for fence in ("```python", "```py", "```"):
            start = response.find(fence)
            if start != -1:
                end = response.find("```", start + len(fence))
                if end != -1:
                    return response[start + len(fence):end].strip()
        lines = response.strip().splitlines()
        code_lines, started = [], False
        for line in lines:
            if not started and line.startswith(
                ("import ", "from ", "def ", "class ", "#", '"""', "'''")
            ):
                started = True
            if started:
                code_lines.append(line)
        return "\n".join(code_lines) if code_lines else response.strip()
