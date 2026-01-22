"""Main orchestrator for the hypothesis generation pipeline."""

from __future__ import annotations

import json
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.coding_agent import CodingAgent
from src.agents.summary_agent import SummaryAgent
from src.execution.jupyter_executor import JupyterExecutor
from src.llm.base import create_provider
from src.memory.conversation import PipelineState
from src.utils.config import Config, DataManifest
from src.utils.logging import get_logger


class Orchestrator:
    """Orchestrates the hypothesis generation and verification pipeline.

    This is the main controller that:
    - Initializes all agents and executors
    - Runs the main hypothesis-verify-analyze loop
    - Handles state persistence and logging
    - Manages graceful shutdown
    """

    def __init__(
        self,
        config: Config,
        manifest: DataManifest,
        output_dir: Path | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Pipeline configuration
            manifest: Data manifest with finding and data paths
            output_dir: Directory for outputs (overrides config)
        """
        self.config = config
        self.manifest = manifest
        self.output_dir = Path(output_dir or config.pipeline.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("orchestrator")
        self._shutdown_requested = False

        # Initialize state
        self.state = PipelineState(
            finding=manifest.finding,
            context=manifest.context,
            max_iterations=config.pipeline.max_iterations,
        )

        # Initialize LLM providers
        self._init_providers()

        # Initialize agents
        self._init_agents()

        # Executor will be initialized when running
        self.executor: JupyterExecutor | None = None

    def _init_providers(self) -> None:
        """Initialize LLM providers based on configuration."""
        self.logger.info("Initializing LLM providers")

        llm_config = self.config.llm

        self.hypothesis_llm = create_provider(
            provider_name=llm_config.hypothesis_model.provider,
            model=llm_config.hypothesis_model.model,
            api_key=llm_config.hypothesis_model.get_api_key(),
            temperature=llm_config.hypothesis_model.temperature,
            max_tokens=llm_config.hypothesis_model.max_tokens,
        )

        self.coding_llm = create_provider(
            provider_name=llm_config.coding_model.provider,
            model=llm_config.coding_model.model,
            api_key=llm_config.coding_model.get_api_key(),
            temperature=llm_config.coding_model.temperature,
            max_tokens=llm_config.coding_model.max_tokens,
        )

        self.summary_llm = create_provider(
            provider_name=llm_config.summary_model.provider,
            model=llm_config.summary_model.model,
            api_key=llm_config.summary_model.get_api_key(),
            temperature=llm_config.summary_model.temperature,
            max_tokens=llm_config.summary_model.max_tokens,
        )

    def _init_agents(self) -> None:
        """Initialize agents with their respective LLM providers."""
        self.logger.info("Initializing agents")

        self.hypothesis_agent = HypothesisAgent(self.hypothesis_llm)
        self.coding_agent = CodingAgent(self.coding_llm)
        self.summary_agent = SummaryAgent(self.summary_llm)

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the pipeline."""
        self.logger.warning("Shutdown requested")
        self._shutdown_requested = True

    def _check_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    async def run(self) -> dict[str, Any]:
        """Run the full hypothesis generation pipeline.

        Returns:
            Dictionary containing final results and report
        """
        self.logger.info("Starting pipeline", {
            "finding": self.state.finding[:100],
            "max_iterations": self.state.max_iterations,
        })

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Start Jupyter executor
            self.executor = JupyterExecutor(
                timeout_seconds=self.config.execution.timeout_seconds,
                working_dir=run_dir,
            )
            await self.executor.start()

            # Generate initial hypotheses
            self.logger.info("Generating initial hypotheses")
            initial_result = await self.hypothesis_agent.generate_initial_hypotheses(
                finding=self.state.finding,
                context=self.state.context,
                data_manifest=self.manifest.model_dump(),
            )

            # Store hypotheses
            for hypo in initial_result.get("hypotheses", []):
                self.state.add_hypothesis(hypo)

            # Save initial state
            self._save_state(run_dir / "state_initial.json")

            # Main loop
            tested_ids: set[int] = set()

            while (
                not self.state.converged
                and self.state.current_iteration < self.state.max_iterations
                and not self._check_shutdown()
            ):
                self.state.current_iteration += 1
                self.logger.info(f"Starting iteration {self.state.current_iteration}")

                # Get next hypothesis to test
                hypothesis = self.hypothesis_agent.get_next_hypothesis(
                    self.state.hypotheses, tested_ids
                )

                if hypothesis is None:
                    self.logger.warning("No more hypotheses to test")
                    break

                hypo_id = hypothesis.get("id", len(tested_ids))
                tested_ids.add(hypo_id)
                self.state.current_hypothesis_index = hypo_id

                # Run verification cycle
                result = await self._run_verification_cycle(
                    hypothesis, run_dir
                )

                # Update state with results
                self.state.tested_hypotheses.append({
                    **hypothesis,
                    "result": result.get("support_level", "UNKNOWN"),
                    "evidence_summary": result.get("summary", ""),
                })

                self.state.add_evidence({
                    "hypothesis_id": hypo_id,
                    "hypothesis_name": hypothesis.get("name", ""),
                    "findings": result.get("findings", []),
                    "support_level": result.get("support_level", "UNKNOWN"),
                    "confidence": result.get("confidence", 0.0),
                    "summary": result.get("summary", ""),
                })

                # Save intermediate state
                if self.config.pipeline.save_intermediate:
                    self._save_state(
                        run_dir / f"state_iter_{self.state.current_iteration}.json"
                    )

                # Check for convergence or refinement
                if not self._check_shutdown():
                    await self._check_and_refine(hypothesis, result)

            # Generate final report
            final_result = await self._generate_final_output(run_dir)

            return final_result

        except Exception as e:
            self.logger.error("Pipeline failed", {"error": str(e)})
            raise

        finally:
            # Cleanup
            if self.executor:
                await self.executor.stop()

    async def _run_verification_cycle(
        self,
        hypothesis: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        """Run a single verification cycle for a hypothesis.

        Args:
            hypothesis: The hypothesis to verify
            run_dir: Directory for outputs

        Returns:
            Result summary from the verification
        """
        self.logger.info("Running verification", {
            "hypothesis": hypothesis.get("name", "N/A"),
        })

        max_retries = self.config.execution.max_retries
        last_code = None
        last_error = None

        for attempt in range(max_retries + 1):
            # Generate code
            code = await self.coding_agent.generate_with_retry(
                hypothesis=hypothesis,
                data_manifest=self.manifest.model_dump(),
                previous_code=last_code,
                previous_error=last_error,
            )

            # Validate code
            is_valid, validation_error = self.coding_agent.validate_code(code)
            if not is_valid:
                self.logger.warning("Code validation failed", {
                    "error": validation_error,
                    "attempt": attempt + 1,
                })
                last_code = code
                last_error = validation_error
                continue

            # Save code
            code_file = run_dir / f"code_iter_{self.state.current_iteration}_attempt_{attempt}.py"
            code_file.write_text(code)

            # Execute code
            self.logger.info("Executing verification code", {
                "attempt": attempt + 1,
            })

            result = await self.executor.execute(code)

            # Save execution output
            output_file = run_dir / f"output_iter_{self.state.current_iteration}_attempt_{attempt}.txt"
            output_file.write_text(result.get_display_output())

            # Save any images
            if result.images:
                await self.executor.save_images(
                    result,
                    run_dir / f"images_iter_{self.state.current_iteration}",
                )

            if result.success:
                # Summarize results
                summary = await self.summary_agent.summarize_results(
                    hypothesis=hypothesis,
                    code=code,
                    execution_result=result,
                )
                return summary
            else:
                self.logger.warning("Execution failed", {
                    "error": result.error,
                    "attempt": attempt + 1,
                })
                last_code = code
                last_error = result.error or "Unknown error"

        # All retries failed
        return {
            "execution_success": False,
            "findings": [],
            "support_level": "ERROR",
            "confidence": 0.0,
            "reasoning": f"Code execution failed after {max_retries + 1} attempts",
            "issues": [last_error],
            "summary": f"Failed to verify hypothesis due to execution errors: {last_error}",
        }

    async def _check_and_refine(
        self,
        last_hypothesis: dict[str, Any],
        last_result: dict[str, Any],
    ) -> None:
        """Check results and potentially refine hypotheses.

        Args:
            last_hypothesis: The hypothesis that was just tested
            last_result: Results from testing
        """
        self.logger.info("Checking results and refining")

        refinement = await self.hypothesis_agent.refine_hypotheses(
            state=self.state,
            last_hypothesis=last_hypothesis,
            last_result=last_result,
            data_manifest=self.manifest.model_dump(),
        )

        decision = refinement.get("decision", "CONTINUE")

        if decision == "CONVERGED":
            self.state.mark_converged(
                reason="Hypothesis confirmed",
                confidence=refinement.get("confidence", 0.8),
                conclusion=refinement.get("conclusion", ""),
            )
        elif decision == "INSUFFICIENT_DATA":
            self.state.mark_converged(
                reason="Insufficient data to continue",
                confidence=refinement.get("confidence", 0.3),
                conclusion=refinement.get("conclusion", ""),
            )
        elif decision in ("REFINE", "NEW_HYPOTHESIS"):
            # Add new hypotheses
            for hypo in refinement.get("hypotheses", []):
                self.state.add_hypothesis(hypo)

    async def _generate_final_output(self, run_dir: Path) -> dict[str, Any]:
        """Generate final output and report.

        Args:
            run_dir: Directory for outputs

        Returns:
            Final result dictionary
        """
        self.logger.info("Generating final output")

        # Format evidence for report
        evidence = self.summary_agent.format_evidence_for_report(self.state.evidence)

        # Generate report
        report = await self.summary_agent.generate_final_report(
            finding=self.state.finding,
            context=self.state.context,
            history_summary=self.state.get_history_summary(),
            conclusion=self.state.conclusion or "Investigation incomplete",
            all_evidence=evidence,
        )

        # Save report
        report_file = run_dir / "final_report.md"
        report_file.write_text(report)

        # Save final state
        self._save_state(run_dir / "state_final.json")

        self.logger.info("Pipeline complete", {
            "converged": self.state.converged,
            "iterations": self.state.current_iteration,
            "report_file": str(report_file),
        })

        return {
            "converged": self.state.converged,
            "convergence_reason": self.state.convergence_reason,
            "confidence": self.state.confidence_level,
            "conclusion": self.state.conclusion,
            "iterations": self.state.current_iteration,
            "report_file": str(report_file),
            "output_dir": str(run_dir),
        }

    def _save_state(self, filepath: Path) -> None:
        """Save pipeline state to file.

        Args:
            filepath: Path to save state
        """
        state_dict = self.state.to_dict()
        filepath.write_text(json.dumps(state_dict, indent=2, default=str))
        self.logger.debug("Saved state", {"path": str(filepath)})
