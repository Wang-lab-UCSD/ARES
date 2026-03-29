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

        system_prompt = build_repl_system_prompt()
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

        for iteration in range(1, max_iterations + 1):
            self.logger.info("REPL iteration", {
                "iteration": iteration,
                "hypothesis": hypothesis.get("name", "N/A"),
            })

            response = await self.llm.complete(messages)
            content = response.content

            # Save LLM turn
            try:
                (debug_dir / f"iter{self.state_iter}_repl{iteration}_llm.txt"
                 ).write_text(content, encoding="utf-8")
            except Exception:
                pass

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
            messages.append(Message.assistant(content))

            # Execute only the FIRST block per LLM turn — true incremental REPL.
            # The model must see each result before writing the next step.
            # Multiple blocks in one response would skip intermediate feedback,
            # causing cascade failures when an early block fails silently.
            code = execute_blocks[0].strip()
            block_idx = 0
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
            "support_level: SUPPORTS|REFUTES|INCONCLUSIVE\n"
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
        valid = {"SUPPORTS", "REFUTES", "INCONCLUSIVE", "ERROR", "UNTESTABLE"}
        level = raw if raw in valid else "INCONCLUSIVE"

        try:
            conf = float(_field("confidence", "0.5"))
            conf = max(0.0, min(1.0, conf))
        except ValueError:
            conf = 0.5

        finding = _field("finding", "")
        reasoning = _field("reasoning", text)
        raw_output = "\n".join(stdout_parts)

        return {
            "execution_success": level not in ("ERROR",),
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
