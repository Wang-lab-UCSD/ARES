"""Code generation agent for verification experiments."""

from __future__ import annotations

import re
from typing import Any

from src.llm.base import LLMProvider, Message
from src.prompts.coding import (
    CODING_SYSTEM_PROMPT,
    build_verification_code_prompt,
    build_error_fix_prompt,
    _format_data_for_coding,
)
from src.utils.logging import get_logger


class CodingAgent:
    """Agent for generating verification code.

    This agent:
    - Generates Python code to verify hypotheses
    - Fixes code errors based on execution feedback
    - Handles retry logic for code generation
    """

    def __init__(self, llm: LLMProvider):
        """Initialize the coding agent.

        Args:
            llm: LLM provider for code generation
        """
        self.llm = llm
        self.logger = get_logger("coding_agent")

    async def generate_verification_code(
        self,
        hypothesis: dict[str, Any],
        data_manifest: dict[str, Any],
        prior_evidence: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate code to verify a hypothesis.

        Args:
            hypothesis: The hypothesis to verify
            data_manifest: Available data paths and tools
            prior_evidence: Evidence from previously tested hypotheses

        Returns:
            Python code as a string
        """
        self.logger.info("Generating verification code", {
            "hypothesis": hypothesis.get("name", "N/A"),
        })

        prompt = build_verification_code_prompt(
            hypothesis, data_manifest, prior_evidence=prior_evidence
        )

        try:
            response = await self.llm.complete(
                [
                    Message.system(CODING_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                # temperature removed for gpt-5-mini compatibility  # Low temperature for more deterministic code
            )

            code = self._extract_code(response.content)

            self.logger.info("Generated code", {"code_length": len(code)})
            return code

        except Exception as e:
            self.logger.error("Failed to generate code", {"error": str(e)})
            raise

    async def fix_code_error(
        self,
        original_code: str,
        error_message: str,
        error_traceback: str,
        data_manifest: dict[str, Any],
        stdout: str = "",
        stderr: str = "",
    ) -> str:
        """Fix code that resulted in an error.

        Args:
            original_code: The code that failed
            error_message: The error message
            error_traceback: Full traceback
            data_manifest: Available data paths
            stdout: Standard output from execution (may contain debug info)
            stderr: Standard error from execution (may contain warnings/errors)

        Returns:
            Fixed Python code as a string
        """
        self.logger.info("Fixing code error", {"error": error_message[:100]})

        prompt = build_error_fix_prompt(
            original_code, error_message, error_traceback, data_manifest,
            stdout=stdout, stderr=stderr,
        )

        try:
            response = await self.llm.complete(
                [
                    Message.system(CODING_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                # temperature removed for gpt-5-mini compatibility
            )

            code = self._extract_code(response.content)

            self.logger.info("Fixed code", {"code_length": len(code)})
            return code

        except Exception as e:
            self.logger.error("Failed to fix code", {"error": str(e)})
            raise

    async def generate_inspection_code(
        self,
        data_manifest: dict[str, Any],
    ) -> str:
        """Generate code to inspect all files in the data manifest.

        Used in the pre-loop file familiarization step (iteration 0).
        No hypothesis involved — just prints file shapes, column names, and samples.

        Args:
            data_manifest: Available data paths

        Returns:
            Python code as a string
        """
        self.logger.info("Generating file inspection code")

        data_section = _format_data_for_coding(data_manifest)

        prompt = f"""# Task: Inspect All Data Files

Generate Python code to inspect every file listed in the data manifest below.

For each file:
1. Print the file path as a header
2. Print the number of lines (or rows for tabular files)
3. If tabular (TSV, CSV, BED, narrowPeak, any whitespace-delimited): print column names and 3 sample rows using pandas
4. If binary (bigWig, bam, hic): print the file size in MB and note the format
5. If FASTA or GTF: print line count and first 3 non-comment lines

Print clearly labeled output per file. Do not perform any analysis or statistics.
Do not import matplotlib or generate any plots.
Use try/except around each file so one failure doesn't stop the rest.

# Available Data

{data_section}
"""

        try:
            response = await self.llm.complete(
                [
                    Message.system(CODING_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
            )

            code = self._extract_code(response.content)
            self.logger.info("Generated inspection code", {"code_length": len(code)})
            return code

        except Exception as e:
            self.logger.error("Failed to generate inspection code", {"error": str(e)})
            raise

    async def generate_with_retry(
        self,
        hypothesis: dict[str, Any],
        data_manifest: dict[str, Any],
        previous_code: str | None = None,
        previous_error: str | None = None,
        previous_stdout: str = "",
        previous_stderr: str = "",
        prior_evidence: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate code with context from previous attempts.

        Args:
            hypothesis: The hypothesis to verify
            data_manifest: Available data paths
            previous_code: Code from previous attempt
            previous_error: Error from previous attempt
            previous_stdout: Stdout from previous attempt (for debugging)
            previous_stderr: Stderr from previous attempt (for debugging)
            prior_evidence: Evidence from previously tested hypotheses

        Returns:
            Python code as a string
        """
        if previous_code and previous_error:
            return await self.fix_code_error(
                previous_code,
                previous_error,
                previous_error,  # Using same for both
                data_manifest,
                stdout=previous_stdout,
                stderr=previous_stderr,
            )
        else:
            return await self.generate_verification_code(
                hypothesis, data_manifest, prior_evidence=prior_evidence
            )

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response.

        Handles responses that may include markdown code blocks.

        Args:
            response: Raw LLM response

        Returns:
            Extracted Python code
        """
        response = response.strip()

        # Try multiple patterns for code block extraction
        # Pattern 1: ```python ... ``` with optional newline
        # Pattern 2: ``` ... ``` generic code block
        patterns = [
            r"```python\s*(.*?)```",  # ```python followed by optional whitespace
            r"```py\s*(.*?)```",       # ```py variant
            r"```\s*(.*?)```",         # Generic code block
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code = "\n\n".join(matches).strip()
                if code:  # Only return if we got non-empty code
                    return code

        # If no code blocks found, check if response looks like raw code
        # (starts with import, def, class, #, or common Python patterns)
        if response and any(
            response.lstrip().startswith(prefix)
            for prefix in ("import ", "from ", "def ", "class ", "#", "\"\"\"", "'''")
        ):
            return response

        # Last resort: return response as-is (might be empty)
        self.logger.warning("Could not extract code block from response", {
            "response_preview": response[:200] if response else "(empty)",
        })
        return response

    def validate_code(self, code: str) -> tuple[bool, str | None]:
        """Perform basic validation of generated code.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code.strip():
            return False, "Empty code"

        # Check for syntax errors
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for potentially dangerous operations
        dangerous_patterns = [
            r"\bos\.system\b",
            r"\beval\b",
            r"\bexec\b",
            r"\b__import__\b",
            r"\brm\s+-rf\b",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                self.logger.warning("Potentially dangerous pattern in code", {
                    "pattern": pattern,
                })
                # Don't fail, just warn - these might be legitimate

        return True, None
