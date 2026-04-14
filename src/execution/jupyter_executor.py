"""Jupyter kernel executor for running Python code."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from jupyter_client import KernelManager
from jupyter_client.kernelspec import NoSuchKernel

from src.utils.logging import get_logger


# Per-channel caps on captured output. Keeps any single execution from
# bloating the REPL observation history and blowing the next LLM call's
# context window. Seen in the wild: the Apr 13 ARID4B/YY1 run crashed when a
# `parse_bedtools_wa_wb(subprocess_stdout)` call triggered a pandas
# OSError whose traceback embedded the entire 3MB stdout, ballooning the
# next MiniMax request from 19K to 138K tokens.
#
# 20 KB per channel ≈ 5K tokens — enough to retain full context for
# normal output while still bounding worst-case growth.
_MAX_STDOUT_BYTES = 20_000
_MAX_STDERR_BYTES = 20_000
_MAX_TRACEBACK_BYTES = 20_000
_TRUNCATION_HEAD_FRACTION = 0.6  # 60% head, 40% tail


def _truncate_output(text: str, max_bytes: int) -> str:
    """Cap a captured output channel to ``max_bytes`` with a head+tail split.

    Preserves the first 60% of the budget from the start of the text (where
    echoed commands and setup usually live) and the last 40% from the end
    (where errors, final results, and the most recent stack frame usually
    live). Inserts a single marker line in between that reports how many
    bytes were dropped so the LLM can see that truncation happened.

    Returns the input unchanged if it already fits.

    Notes:
    - Byte-length is measured via ``len(text)`` (Python str == code points,
      which is sufficient for the ASCII-dominated output we see in practice).
    - ``max_bytes`` must be large enough to fit the marker; values < 200
      fall back to "just return the tail" to avoid degenerate slices.
    """
    if max_bytes <= 0 or len(text) <= max_bytes:
        return text

    original_len = len(text)

    # Degenerate-tiny budget: just keep the tail (most recent output).
    if max_bytes < 200:
        return text[-max_bytes:]

    marker = (
        f"\n... [truncated {original_len - max_bytes:,} of {original_len:,} bytes; "
        f"showing first {int(max_bytes * _TRUNCATION_HEAD_FRACTION):,} + last "
        f"{max_bytes - int(max_bytes * _TRUNCATION_HEAD_FRACTION):,} bytes] ...\n"
    )

    head_budget = int(max_bytes * _TRUNCATION_HEAD_FRACTION)
    tail_budget = max_bytes - head_budget

    return text[:head_budget] + marker + text[-tail_budget:]


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    error_traceback: str | None = None
    outputs: list[dict[str, Any]] = field(default_factory=list)
    images: list[bytes] = field(default_factory=list)
    execution_count: int = 0

    @staticmethod
    def _normalize_line(line: str) -> str:
        """Normalize a line for dedup by replacing variable parts with placeholders."""
        import re
        # Replace motif IDs like MA0138.2, +MA0138.2, -MA0138.2
        s = re.sub(r'[+-]?MA\d+\.\d+\.?', '<MOTIF>', line)
        # Replace file paths
        s = re.sub(r'/[\w/._-]+\.\w+', '<PATH>', s)
        # Replace numbers
        s = re.sub(r'\b\d+\b', '<N>', s)
        return s

    @staticmethod
    def _filter_noisy_output(text: str, max_repeats: int = 5) -> str:
        """Filter repetitive lines from output (e.g., FIMO 'Skipping motif' spam).

        Lines are normalized (variable parts like motif IDs replaced) before
        counting, so 'Skipping motif +MA0138.2' and 'Skipping motif +MA0139.1'
        count as the same pattern. Keeps the first `max_repeats` occurrences
        and replaces the rest with a summary.
        """
        counts: dict[str, int] = {}
        filtered: list[str] = []
        for line in text.splitlines():
            key = ExecutionResult._normalize_line(line)
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= max_repeats:
                filtered.append(line)
            elif counts[key] == max_repeats + 1:
                filtered.append("  ... (repeated pattern, suppressed further occurrences)")

        suppressed = {k: n for k, n in counts.items() if n > max_repeats}
        if suppressed:
            total_suppressed = sum(n - max_repeats for n in suppressed.values())
            filtered.append(
                f"\n[Suppressed {len(suppressed)} repeated line pattern(s), "
                f"{total_suppressed} lines removed]"
            )

        return "\n".join(filtered)

    def get_display_output(self) -> str:
        """Get a human-readable summary of the execution."""
        parts = []

        if self.stdout:
            parts.append(f"=== STDOUT ===\n{self.stdout}")

        if self.stderr:
            filtered_stderr = self._filter_noisy_output(self.stderr)
            parts.append(f"=== STDERR ===\n{filtered_stderr}")

        if self.error:
            parts.append(f"=== ERROR ===\n{self.error}")
            if self.error_traceback:
                parts.append(f"Traceback:\n{self.error_traceback}")

        for i, output in enumerate(self.outputs):
            if "text/plain" in output:
                parts.append(f"=== Output {i+1} ===\n{output['text/plain']}")

        if self.images:
            parts.append(f"=== Images ===\n{len(self.images)} image(s) generated")

        return "\n\n".join(parts) if parts else "(No output)"


class JupyterExecutor:
    """Manages a Jupyter kernel for executing Python code.

    Features:
    - Persistent kernel session (variables persist between executions)
    - Captures stdout, stderr, and display outputs
    - Supports rich outputs (images, HTML, etc.)
    - Timeout handling
    - Can execute subprocess calls for CLI tools
    """

    def __init__(
        self,
        timeout_seconds: int = 300,
        working_dir: str | Path | None = None,
        kernel_name: str = "python3",
        max_start_attempts: int = 3,
    ):
        """Initialize the executor.

        Args:
            timeout_seconds: Maximum time for code execution
            working_dir: Working directory for the kernel
            kernel_name: Jupyter kernel to use
            max_start_attempts: How many times to retry kernel startup on
                transient failures (ZMQ port collision, bootstrap timeout).
                Each attempt kills the previous kernel before retrying.
        """
        self.timeout_seconds = timeout_seconds
        self.working_dir = Path(working_dir).resolve() if working_dir else Path.cwd()
        self.kernel_name = kernel_name
        self.max_start_attempts = max_start_attempts

        self.logger = get_logger("jupyter")
        self._km: KernelManager | None = None
        self._kc = None  # Kernel client
        self._execution_count = 0
        self._consecutive_timeouts = 0
        self.on_restart: Callable[[], Awaitable[None]] | None = None
        self.project_root = Path(__file__).resolve().parents[2]

    def _build_kernel_bootstrap_code(self) -> str:
        """Return code that makes project-local imports work inside the kernel."""
        project_root = json.dumps(str(self.project_root))
        working_dir = json.dumps(str(self.working_dir))
        return f"""
import os
import sys
from pathlib import Path

_codex_project_root = {project_root}
_codex_working_dir = {working_dir}
if _codex_project_root not in sys.path:
    sys.path.insert(0, _codex_project_root)
os.environ["PYTHONPATH"] = _codex_project_root + os.pathsep + os.environ.get("PYTHONPATH", "")
Path(_codex_working_dir).mkdir(parents=True, exist_ok=True)
os.chdir(_codex_working_dir)

import pandas as pd
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 120)
pd.set_option('display.max_colwidth', 40)
""".strip()

    async def start(self) -> None:
        """Start the Jupyter kernel, retrying on transient startup failures.

        Transient failures (ZMQ port collision, bootstrap timeout) are retried
        up to `max_start_attempts` times. Each failed attempt kills the hung
        kernel before spawning a fresh one. `NoSuchKernel` is not retried.
        """
        if self._km is not None:
            self.logger.warning("Kernel already running")
            return

        self.logger.info("Starting Jupyter kernel", {
            "kernel": self.kernel_name,
            "working_dir": str(self.working_dir),
        })

        last_error: Exception | None = None
        for attempt in range(1, self.max_start_attempts + 1):
            if attempt > 1:
                self.logger.warning(
                    "Retrying kernel startup",
                    {"attempt": attempt, "max_attempts": self.max_start_attempts,
                     "reason": str(last_error)},
                )
                await asyncio.sleep(2)

            try:
                self._km = KernelManager(kernel_name=self.kernel_name)
                self._km.start_kernel(cwd=str(self.working_dir))
                self._kc = self._km.client()
                self._kc.start_channels()

                await self._wait_for_ready()

                # Bootstrap code is trivial (sys.path + os.chdir); use a short
                # timeout so a ZMQ hang is detected quickly and retried.
                orig_timeout = self.timeout_seconds
                self.timeout_seconds = 60
                try:
                    bootstrap_result = await self.execute(self._build_kernel_bootstrap_code())
                finally:
                    self.timeout_seconds = orig_timeout

                if not bootstrap_result.success:
                    raise RuntimeError(
                        "Failed to initialize Jupyter kernel import path: "
                        f"{bootstrap_result.error or 'unknown bootstrap error'}"
                    )

                self.logger.info("Kernel started successfully")
                return

            except NoSuchKernel:
                # Permanent — no point retrying
                self.logger.error("Kernel not found", {"kernel": self.kernel_name})
                raise RuntimeError(
                    f"Kernel '{self.kernel_name}' not found. Is ipykernel installed?"
                )
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "Kernel startup attempt failed — killing kernel before retry",
                    {"attempt": attempt, "error": str(e)},
                )
                # Kill the hung kernel so the next attempt starts completely fresh
                await self._force_stop()

        self.logger.error("Failed to start kernel after all attempts", {
            "max_attempts": self.max_start_attempts,
            "error": str(last_error),
        })
        raise RuntimeError(
            f"Failed to start kernel after {self.max_start_attempts} attempts: {last_error}"
        )

    async def _force_stop(self) -> None:
        """Kill the kernel unconditionally, ignoring all errors. Used between retries."""
        try:
            if self._kc:
                self._kc.stop_channels()
        except Exception:
            pass
        try:
            if self._km:
                self._km.shutdown_kernel(now=True)
        except Exception:
            pass
        finally:
            self._km = None
            self._kc = None

    async def _wait_for_ready(self, timeout: float = 30.0) -> None:
        """Wait for the kernel to be ready."""
        if not self._kc:
            raise RuntimeError("Kernel client not initialized")

        start_time = asyncio.get_event_loop().time()
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError("Kernel failed to become ready")

            try:
                self._kc.kernel_info()
                # Try to get a response
                msg = self._kc.get_shell_msg(timeout=1)
                if msg["msg_type"] == "kernel_info_reply":
                    break
            except queue.Empty:
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop the Jupyter kernel."""
        if self._km is None:
            return

        self.logger.info("Stopping Jupyter kernel")

        try:
            if self._kc:
                self._kc.stop_channels()
            self._km.shutdown_kernel(now=True)
        except Exception as e:
            self.logger.warning("Error stopping kernel", {"error": str(e)})
        finally:
            self._km = None
            self._kc = None

    async def execute(
        self,
        code: str,
        *,
        skip_truncation: bool = False,
    ) -> ExecutionResult:
        """Execute Python code in the kernel.

        Args:
            code: Python code to execute
            skip_truncation: If True, return the full captured stdout/stderr/
                traceback with no size cap. Use for one-shot inspection calls
                where the output is stored persistently (e.g. file
                familiarization at iter 0). The default REPL path keeps the
                cap so a single runaway print doesn't blow the next LLM call's
                context window.

        Returns:
            ExecutionResult with stdout, stderr, outputs, etc.
        """
        if self._kc is None:
            await self.start()

        self._execution_count += 1
        self.logger.debug("Executing code", {
            "execution_count": self._execution_count,
            "code_length": len(code),
        })

        # Execute the code
        msg_id = self._kc.execute(code)

        # Collect outputs
        stdout_parts = []
        stderr_parts = []
        outputs = []
        images = []
        error = None
        error_traceback = None

        # Process messages until execution is complete
        try:
            while True:
                try:
                    # timeout=None means wait forever; 0 is treated as no timeout
                    timeout = None if self.timeout_seconds == 0 else self.timeout_seconds
                    msg = self._kc.get_iopub_msg(timeout=timeout)
                except queue.Empty:
                    self.logger.error("Execution timed out")
                    # Interrupt the kernel to kill the hung execution so the
                    # next code block can run on a clean kernel.
                    try:
                        if self._km:
                            self._km.interrupt_kernel()
                            self.logger.info("Kernel interrupted after timeout")
                    except Exception as intr_err:
                        self.logger.warning(
                            "Failed to interrupt kernel after timeout",
                            {"error": str(intr_err)},
                        )
                    self._consecutive_timeouts += 1
                    # After a timeout, verify the kernel actually recovered by
                    # running a trivial canary. If the canary also hangs, the
                    # kernel is unrecoverable — force-restart it. Without this
                    # check, past runs have burned 2+ hours sending 1+1 to a
                    # dead kernel that keeps returning "timed out" for every
                    # subsequent execution.
                    if await self._kernel_is_dead():
                        self.logger.warning(
                            "Kernel is unresponsive after interrupt — force-restarting",
                            {"consecutive_timeouts": self._consecutive_timeouts},
                        )
                        await self._force_restart_after_death()
                    elif self._consecutive_timeouts >= 2:
                        # Canary passed but we've timed out twice in a row —
                        # the kernel is probably accumulating broken state.
                        # Restart proactively.
                        self.logger.warning(
                            "Two consecutive timeouts — restarting kernel proactively",
                            {"consecutive_timeouts": self._consecutive_timeouts},
                        )
                        await self._force_restart_after_death()
                    return ExecutionResult(
                        success=False,
                        error="Execution timed out",
                        execution_count=self._execution_count,
                    )

                msg_type = msg["msg_type"]
                content = msg.get("content", {})

                # Check if this message is for our execution
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                if msg_type == "stream":
                    name = content.get("name", "")
                    text = content.get("text", "")
                    if name == "stdout":
                        stdout_parts.append(text)
                    elif name == "stderr":
                        stderr_parts.append(text)

                elif msg_type == "execute_result":
                    data = content.get("data", {})
                    outputs.append(data)

                elif msg_type == "display_data":
                    data = content.get("data", {})
                    outputs.append(data)

                    # Extract images
                    if "image/png" in data:
                        img_data = base64.b64decode(data["image/png"])
                        images.append(img_data)
                    elif "image/jpeg" in data:
                        img_data = base64.b64decode(data["image/jpeg"])
                        images.append(img_data)

                elif msg_type == "error":
                    error = content.get("ename", "Error")
                    error_value = content.get("evalue", "")
                    traceback_lines = content.get("traceback", [])
                    error = f"{error}: {error_value}"
                    error_traceback = "\n".join(traceback_lines)

                elif msg_type == "status":
                    if content.get("execution_state") == "idle":
                        # Execution complete
                        break

        except Exception as e:
            self.logger.error("Error during execution", {"error": str(e)})
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_count=self._execution_count,
            )

        success = error is None
        # Execution returned — kernel is alive, reset the consecutive-timeout counter.
        self._consecutive_timeouts = 0

        # Cap each captured channel before persisting into ExecutionResult.
        # See note at the top of the module for the Apr 13 ARID4B/YY1 crash
        # that motivated this: a single 3MB pandas traceback blew the next
        # MiniMax call's context window (19K → 138K tokens in one step).
        # skip_truncation=True preserves full output for one-shot callers
        # (e.g. file familiarization) that store results persistently.
        raw_stdout = "".join(stdout_parts)
        raw_stderr = "".join(stderr_parts)
        if skip_truncation:
            final_stdout = raw_stdout
            final_stderr = raw_stderr
            final_traceback = error_traceback
        else:
            final_stdout = _truncate_output(raw_stdout, _MAX_STDOUT_BYTES)
            final_stderr = _truncate_output(raw_stderr, _MAX_STDERR_BYTES)
            final_traceback = (
                _truncate_output(error_traceback, _MAX_TRACEBACK_BYTES)
                if error_traceback is not None
                else None
            )

        result = ExecutionResult(
            success=success,
            stdout=final_stdout,
            stderr=final_stderr,
            error=error,
            error_traceback=final_traceback,
            outputs=outputs,
            images=images,
            execution_count=self._execution_count,
        )

        self.logger.debug("Execution complete", {
            "success": success,
            "stdout_length": len(result.stdout),
            "stdout_raw_length": len(raw_stdout),
            "stderr_length": len(result.stderr),
            "stderr_raw_length": len(raw_stderr),
            "output_count": len(outputs),
            "image_count": len(images),
        })

        return result

    async def _kernel_is_dead(self, canary_timeout: float = 5.0) -> bool:
        """Run a trivial canary command to check if the kernel is responsive.

        Returns True if the canary times out or errors — meaning the kernel
        is stuck or dead and cannot execute even the simplest expression.
        Returns False if the canary succeeds.

        Note: this method sends directly to the kernel client and does NOT
        recursively call execute() — doing so would loop on dead kernels.
        """
        if self._kc is None or self._km is None:
            return True
        try:
            msg_id = self._kc.execute("1+1")
            deadline = asyncio.get_event_loop().time() + canary_timeout
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    return True
                try:
                    msg = self._kc.get_iopub_msg(timeout=min(remaining, 1.0))
                except queue.Empty:
                    continue
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                if msg.get("msg_type") == "status" and msg.get("content", {}).get("execution_state") == "idle":
                    return False  # canary completed — kernel is alive
        except Exception as e:
            self.logger.warning("Canary check failed with exception", {"error": str(e)})
            return True

    async def _force_restart_after_death(self) -> None:
        """Kill the current kernel and start a fresh one.

        Calls the on_restart callback (if set) so the orchestrator can
        re-inject data_files and any other required kernel state.
        """
        try:
            await self._force_stop()
        except Exception as e:
            self.logger.warning("Error during force-stop", {"error": str(e)})
        self._km = None
        self._kc = None
        self._execution_count = 0
        self._consecutive_timeouts = 0
        try:
            await self.start()
            self.logger.info("Kernel force-restarted after unresponsiveness")
        except Exception as e:
            self.logger.error("Failed to restart kernel after death", {"error": str(e)})
            return
        # Let the orchestrator re-inject data_files and any other state
        if self.on_restart is not None:
            try:
                await self.on_restart()
                self.logger.info("on_restart callback completed — kernel state restored")
            except Exception as e:
                self.logger.error("on_restart callback failed", {"error": str(e)})

    async def execute_with_retry(
        self,
        code: str,
        max_retries: int = 3,
    ) -> ExecutionResult:
        """Execute code with retry on failure.

        Args:
            code: Python code to execute
            max_retries: Maximum retry attempts

        Returns:
            ExecutionResult from the last attempt
        """
        last_result = None

        for attempt in range(max_retries):
            result = await self.execute(code)

            if result.success:
                return result

            last_result = result
            self.logger.warning("Execution failed, retrying", {
                "attempt": attempt + 1,
                "max_retries": max_retries,
                "error": result.error,
            })

        return last_result or ExecutionResult(
            success=False,
            error="Max retries exceeded",
            execution_count=self._execution_count,
        )

    async def save_images(self, result: ExecutionResult, output_dir: Path) -> list[Path]:
        """Save images from execution result to files.

        Args:
            result: ExecutionResult containing images
            output_dir: Directory to save images

        Returns:
            List of paths to saved images
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for i, img_data in enumerate(result.images):
            filename = f"output_{result.execution_count}_{i}.png"
            filepath = output_dir / filename
            filepath.write_bytes(img_data)
            saved.append(filepath)
            self.logger.debug("Saved image", {"path": str(filepath)})

        return saved

    async def __aenter__(self) -> JupyterExecutor:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
