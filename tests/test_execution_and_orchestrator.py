"""Regression tests for execution bootstrap and orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.execution.jupyter_executor import JupyterExecutor
from src.orchestrator import Orchestrator


def test_jupyter_bootstrap_adds_project_root_and_workdir(tmp_path):
    """The kernel bootstrap should make local src imports work reliably."""
    executor = JupyterExecutor(working_dir=tmp_path)

    code = executor._build_kernel_bootstrap_code()

    assert "sys.path.insert(0, _codex_project_root)" in code
    assert 'os.environ["PYTHONPATH"]' in code
    assert str(Path(__file__).resolve().parents[1]) in code
    assert str(tmp_path) in code


def test_verification_cycle_delegates_to_run_repl(tmp_path):
    """_run_verification_cycle should call coding_agent.run_repl and return its result."""

    expected_result = {
        "support_level": "SUPPORTS",
        "confidence": 0.9,
        "reasoning": "p < 0.05",
        "_raw_output": "some output",
    }

    class StubCodingAgent:
        state_iter = 0

        async def run_repl(self, **kwargs):
            return expected_result

    class StubExecutor:
        """Minimal executor stub — _run_verification_cycle re-injects
        data_files at the start of every cycle, which needs .execute()."""
        async def execute(self, code):
            return SimpleNamespace(success=True, stderr="")

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    orchestrator.config = SimpleNamespace(execution=SimpleNamespace(max_retries=5))
    orchestrator.manifest = SimpleNamespace(model_dump=lambda: {})
    orchestrator.state = SimpleNamespace(
        evidence=[], file_summaries={}, tested_hypotheses=[], current_iteration=1
    )
    orchestrator.coding_agent = StubCodingAgent()
    orchestrator._allowed_packages = []
    orchestrator.executor = StubExecutor()

    result = asyncio.run(
        orchestrator._run_verification_cycle(
            {"name": "test hypothesis", "prediction": "", "verification_plan": []},
            tmp_path,
        )
    )

    assert result["support_level"] == "SUPPORTS"
    assert orchestrator.coding_agent.state_iter == 1  # set from state.current_iteration
