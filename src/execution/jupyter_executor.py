"""Jupyter kernel executor for running Python code."""

from __future__ import annotations

import asyncio
import base64
import os
import queue
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jupyter_client import KernelManager
from jupyter_client.kernelspec import NoSuchKernel

from src.utils.logging import get_logger


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

    def get_display_output(self) -> str:
        """Get a human-readable summary of the execution."""
        parts = []

        if self.stdout:
            parts.append(f"=== STDOUT ===\n{self.stdout}")

        if self.stderr:
            parts.append(f"=== STDERR ===\n{self.stderr}")

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
    ):
        """Initialize the executor.

        Args:
            timeout_seconds: Maximum time for code execution
            working_dir: Working directory for the kernel
            kernel_name: Jupyter kernel to use
        """
        self.timeout_seconds = timeout_seconds
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.kernel_name = kernel_name

        self.logger = get_logger("jupyter")
        self._km: KernelManager | None = None
        self._kc = None  # Kernel client
        self._execution_count = 0

    async def start(self) -> None:
        """Start the Jupyter kernel."""
        if self._km is not None:
            self.logger.warning("Kernel already running")
            return

        self.logger.info("Starting Jupyter kernel", {
            "kernel": self.kernel_name,
            "working_dir": str(self.working_dir),
        })

        try:
            self._km = KernelManager(kernel_name=self.kernel_name)
            self._km.start_kernel(cwd=str(self.working_dir))
            self._kc = self._km.client()
            self._kc.start_channels()

            # Wait for kernel to be ready
            await self._wait_for_ready()

            self.logger.info("Kernel started successfully")

        except NoSuchKernel:
            self.logger.error("Kernel not found", {"kernel": self.kernel_name})
            raise RuntimeError(f"Kernel '{self.kernel_name}' not found. Is ipykernel installed?")
        except Exception as e:
            self.logger.error("Failed to start kernel", {"error": str(e)})
            raise

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

    async def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in the kernel.

        Args:
            code: Python code to execute

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
        result = ExecutionResult(
            success=success,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            error=error,
            error_traceback=error_traceback,
            outputs=outputs,
            images=images,
            execution_count=self._execution_count,
        )

        self.logger.debug("Execution complete", {
            "success": success,
            "stdout_length": len(result.stdout),
            "stderr_length": len(result.stderr),
            "output_count": len(outputs),
            "image_count": len(images),
        })

        return result

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
