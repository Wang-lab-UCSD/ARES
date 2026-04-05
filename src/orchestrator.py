"""Main orchestrator for the hypothesis generation pipeline."""

from __future__ import annotations

import json
import os
import re
import signal
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.coding_agent import CodingAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.review_agent import ReviewAgent
from src.execution.jupyter_executor import JupyterExecutor
from src.llm.base import create_provider
from src.memory.conversation import PipelineState
from src.ui.approval import ApprovalUI, ApprovalDecision, ApprovalResult
from src.utils.config import (
    Config,
    DataManifest,
    DEFAULT_EXECUTION_PACKAGES,
    parse_requirements_package_names,
)
from src.utils.cost_tracker import CostTracker, CostLimitExceeded, SessionBudgetExceeded, TokenLimitExceeded
from src.utils.logging import get_logger
from src.utils.narrative import NarrativeLog


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
        manifest_path: Path | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Pipeline configuration
            manifest: Data manifest with finding and data paths
            output_dir: Directory for outputs (overrides config)
            manifest_path: Path to the manifest file (used to resolve relative execution_requirements_path)
        """
        self.config = config
        self.manifest = manifest
        self.output_dir = Path(output_dir or config.pipeline.output_dir)
        self.logger = get_logger("orchestrator")
        self._allowed_packages = self._resolve_allowed_packages(manifest_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._shutdown_requested = False
        self._preflight_check(manifest_path)

        # Initialize state (include investigation_objective in context when set)
        context = manifest.context
        if manifest.investigation_objective and manifest.investigation_objective.strip():
            context = (context.rstrip() + "\n\nInvestigation objective: " + manifest.investigation_objective.strip()).strip()
        self.state = PipelineState(
            finding=manifest.finding,
            context=context,
            max_iterations=config.pipeline.max_iterations,
        )

        # Initialize cost tracker if enabled
        self.cost_tracker: CostTracker | None = None
        if config.cost.enabled:
            self.cost_tracker = CostTracker(
                per_call_limit=config.cost.per_call_limit_usd,
                session_limit=config.cost.session_limit_usd,
                warn_threshold=config.cost.warn_threshold,
            )
            self.logger.info("Cost tracking enabled", {
                "per_call_limit_usd": config.cost.per_call_limit_usd,
                "session_limit_usd": config.cost.session_limit_usd,
            })

        # Initialize approval UI if interactive mode enabled
        self.approval_ui: ApprovalUI | None = None
        if config.interactive.enabled:
            self.approval_ui = ApprovalUI(use_rich=config.interactive.use_rich)
            self.logger.info("Interactive approval mode enabled")

        # Initialize LLM providers
        self._init_providers()

        # Initialize agents
        self._init_agents()

        # Executor will be initialized when running
        self.executor: JupyterExecutor | None = None

    @staticmethod
    def _iter_strings(obj: Any):
        """Yield all string leaf values from an arbitrarily nested structure."""
        if isinstance(obj, dict):
            for v in obj.values():
                yield from Orchestrator._iter_strings(v)
        elif isinstance(obj, list):
            for item in obj:
                yield from Orchestrator._iter_strings(item)
        elif isinstance(obj, str):
            yield obj

    def _preflight_check(self, manifest_path: Path | None) -> None:
        """Fail fast if required CLI tools or critical manifest data are missing.

        This avoids expensive LLM calls + retries when the runtime environment cannot
        actually run the generated code.
        """
        base_dir = Path(manifest_path).parent if manifest_path else Path.cwd()

        # 1) Validate CLI tools listed in the manifest.
        missing_tools: list[str] = []
        for tool in (self.manifest.tools or []):
            if not tool:
                continue
            if shutil.which(tool) is None:
                missing_tools.append(tool)

        # 2) Validate likely data files referenced in the manifest.
        # Heuristic: only check absolute paths or relative paths that look like
        # filesystem locations, and only file extensions we expect for genomic inputs.
        allowed_exts = {
            ".bed",
            ".bedpe",
            ".bw",
            ".bigwig",
            ".fa",
            ".fasta",
            ".gtf",
            ".tsv",
            ".csv",
            ".meme",
            ".txt",
            ".narrowpeak",
            ".narrowpeak.gz",
            ".phyloP100way.bw",
        }

        missing_files: list[str] = []
        data_section = self.manifest.model_dump().get("data", {}) or {}
        for s in self._iter_strings(data_section):
            # Skip non-path strings and metadata.
            s_stripped = s.strip()
            if not s_stripped:
                continue

            is_probably_path = s_stripped.startswith("/") or s_stripped.startswith(".") or s_stripped.startswith("..")
            if not is_probably_path:
                continue

            # Resolve relative paths against manifest directory.
            candidate = Path(s_stripped)
            if not candidate.is_absolute():
                candidate = base_dir / candidate

            # Only check files with recognized extensions (cache dirs typically have none).
            ext = candidate.name.lower()
            # Special-case: some files include dots in the stem (e.g., .phyloP100way.bw)
            if candidate.suffix.lower() == ".bw":
                normalized = ".bw"
            else:
                normalized = candidate.suffix.lower()
            if normalized not in allowed_exts and ext not in allowed_exts:
                continue

            if not candidate.exists():
                missing_files.append(str(candidate))

        if missing_tools or missing_files:
            parts: list[str] = []
            if missing_tools:
                parts.append(
                    "Missing CLI tools (install in your environment before running): " + ", ".join(sorted(set(missing_tools)))
                )
            if missing_files:
                # Keep message readable; list up to 20.
                head = sorted(set(missing_files))[:20]
                parts.append(
                    "Missing manifest data files (check paths in data_manifest.yaml): "
                    + "; ".join(head)
                    + ("; ..." if len(set(missing_files)) > 20 else "")
                )
            raise RuntimeError("Preflight check failed. " + " ".join(parts))

    def _infer_tools_from_text(self, text: str) -> list[str]:
        """Infer which CLI tool(s) an issue/error likely refers to."""
        if not text:
            return []
        tools = set((self.manifest.tools or []))
        # Always consider common CLIs even if not listed explicitly
        tools.update({"bedtools", "fimo", "meme", "samtools"})
        found = []
        t_low = text.lower()
        for tool in sorted(tools):
            if not tool:
                continue
            if re.search(rf"\b{re.escape(tool.lower())}\b", t_low):
                found.append(tool.lower())
        return sorted(set(found))

    @staticmethod
    def _extract_cli_tools_from_code(code: str) -> list[str]:
        """Heuristically extract CLI tool names from generated code."""
        if not code:
            return []
        # Matches subprocess.run(["tool", ...]) / run_cmd(["tool", ...]) / subprocess.check_output([...])
        tools = re.findall(
            r"(?:subprocess\.(?:run|check_output|check_call)\s*\(\s*\[\s*['\"]([^'\"]+)['\"])|"
            r"(?:run_cmd\s*\(\s*\[\s*['\"]([^'\"]+)['\"])",
            code,
        )
        flat = []
        for a, b in tools:
            if a:
                flat.append(a)
            if b:
                flat.append(b)
        return sorted({t.strip().lower() for t in flat if t and t.strip()})

    @staticmethod
    def _extract_good_invocation_snippets(code: str, tool: str) -> list[str]:
        """Extract short invocation examples for a specific tool from code."""
        if not code or not tool:
            return []
        # Find list-literal subprocess calls that start with the tool
        pattern = rf"(subprocess\.(?:run|check_output|check_call)\s*\(\s*\[\s*['\"]{ re.escape(tool) }['\"][^\]]*\])"
        matches = re.findall(pattern, code)
        snippets = []
        for m in matches[:2]:
            s = " ".join(m.strip().split())
            if len(s) > 220:
                s = s[:217] + "..."
            snippets.append(s)
        return snippets

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
            base_url=llm_config.hypothesis_model.base_url,
            reasoning_effort=llm_config.hypothesis_model.reasoning_effort,
        )

        self.coding_llm = create_provider(
            provider_name=llm_config.coding_model.provider,
            model=llm_config.coding_model.model,
            api_key=llm_config.coding_model.get_api_key(),
            temperature=llm_config.coding_model.temperature,
            max_tokens=llm_config.coding_model.max_tokens,
            base_url=llm_config.coding_model.base_url,
            reasoning_effort=llm_config.coding_model.reasoning_effort,
        )

        self.summary_llm = create_provider(
            provider_name=llm_config.summary_model.provider,
            model=llm_config.summary_model.model,
            api_key=llm_config.summary_model.get_api_key(),
            temperature=llm_config.summary_model.temperature,
            max_tokens=llm_config.summary_model.max_tokens,
            base_url=llm_config.summary_model.base_url,
            reasoning_effort=llm_config.summary_model.reasoning_effort,
        )

        if llm_config.review_model is not None:
            self.review_llm = create_provider(
                provider_name=llm_config.review_model.provider,
                model=llm_config.review_model.model,
                api_key=llm_config.review_model.get_api_key(),
                temperature=llm_config.review_model.temperature,
                max_tokens=llm_config.review_model.max_tokens,
                base_url=llm_config.review_model.base_url,
                reasoning_effort=llm_config.review_model.reasoning_effort,
            )
        else:
            self.review_llm = self.coding_llm

        # Inject cost tracker into all providers
        if self.cost_tracker is not None:
            self.hypothesis_llm.set_cost_tracker(self.cost_tracker)
            self.coding_llm.set_cost_tracker(self.cost_tracker)
            self.summary_llm.set_cost_tracker(self.cost_tracker)
            if llm_config.review_model is not None:
                self.review_llm.set_cost_tracker(self.cost_tracker)

    def _init_agents(self) -> None:
        """Initialize agents with their respective LLM providers."""
        self.logger.info("Initializing agents")

        self.hypothesis_agent = HypothesisAgent(self.hypothesis_llm)
        self.coding_agent = CodingAgent(self.coding_llm)
        self.review_agent = ReviewAgent(self.review_llm)
        self.summary_agent = SummaryAgent(self.summary_llm)

    def _resolve_allowed_packages(self, manifest_path: Path | None) -> list[str]:
        """Resolve allowed Python packages from manifest or default."""
        req_path = getattr(self.manifest, "execution_requirements_path", None)
        if not req_path or not manifest_path:
            return DEFAULT_EXECUTION_PACKAGES
        resolved = Path(manifest_path).parent / req_path
        if not resolved.exists():
            self.logger.warning("Execution requirements file not found, using default packages", {
                "path": str(resolved),
            })
            return DEFAULT_EXECUTION_PACKAGES
        names = parse_requirements_package_names(resolved)
        if names:
            self.logger.info("Using allowed packages from requirements file", {
                "path": str(resolved),
                "count": len(names),
            })
            return names
        return DEFAULT_EXECUTION_PACKAGES

    async def _approve_hypothesis(
        self,
        hypothesis: dict[str, Any],
        iteration: int,
    ) -> ApprovalResult:
        """Request approval for a hypothesis.

        Args:
            hypothesis: The hypothesis to approve
            iteration: Current iteration number

        Returns:
            ApprovalResult with decision and optional feedback
        """
        if self.approval_ui is None:
            return ApprovalResult(ApprovalDecision.APPROVE)

        if not self.config.interactive.approve_hypotheses:
            return ApprovalResult(ApprovalDecision.APPROVE)

        # Display cost summary if tracking
        if self.cost_tracker is not None:
            self.approval_ui.display_cost_summary(self.cost_tracker.get_summary())

        self.approval_ui.display_hypothesis(hypothesis, iteration)
        return self.approval_ui.request_approval("hypothesis")

    async def _approve_code(
        self,
        code: str,
        hypothesis_name: str,
    ) -> ApprovalResult:
        """Request approval for generated code.

        Args:
            code: The generated code to approve
            hypothesis_name: Name of the hypothesis being tested

        Returns:
            ApprovalResult with decision and optional feedback
        """
        if self.approval_ui is None:
            return ApprovalResult(ApprovalDecision.APPROVE)

        if not self.config.interactive.approve_code:
            return ApprovalResult(ApprovalDecision.APPROVE)

        # Display cost summary if tracking
        if self.cost_tracker is not None:
            self.approval_ui.display_cost_summary(self.cost_tracker.get_summary())

        self.approval_ui.display_code(code, hypothesis_name)
        return self.approval_ui.request_approval("code")

    def _display_message(self, message: str, style: str = "info") -> None:
        """Display a message to the user.

        Args:
            message: Message to display
            style: Style hint (info, warning, error, success)
        """
        if self.approval_ui is not None:
            self.approval_ui.display_message(message, style)
        else:
            self.logger.info(message)

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the pipeline."""
        self.logger.warning("Shutdown requested")
        self._shutdown_requested = True

    def _check_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    @staticmethod
    def _flatten_manifest_keys(data_manifest: dict[str, Any]) -> set[str]:
        """Extract all recognized data key forms from the manifest."""
        keys: set[str] = set()
        data = data_manifest.get("data", {})
        for category, items in data.items():
            keys.add(category)
            if isinstance(items, dict):
                for leaf_key in items:
                    keys.add(leaf_key)
                    keys.add(f"{category}.{leaf_key}")
                    keys.add(f"{category}:{leaf_key}")
        return keys

    def _check_data_availability(
        self, hypothesis: dict[str, Any],
    ) -> tuple[bool, list[str], list[str]]:
        """Check if required_data keys exist in the manifest."""
        required = hypothesis.get("required_data", [])
        if not required:
            return True, [], []
        manifest_keys = self._flatten_manifest_keys(self.manifest.model_dump())
        missing = [k for k in required if k not in manifest_keys]
        return len(missing) == 0, missing, sorted(manifest_keys)

    async def run(self) -> dict[str, Any]:
        """Run the full hypothesis generation pipeline.

        Returns:
            Dictionary containing final results and report
        """
        self.logger.info("Starting pipeline", {
            "finding": self.state.finding[:100],
            "max_iterations": self.state.max_iterations,
        })

        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid() % 10000:04d}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        narrative = NarrativeLog(run_dir)
        narrative.write_header(
            finding=self.state.finding,
            run_id=run_id,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

        try:
            # Start Jupyter executor
            self.executor = JupyterExecutor(
                timeout_seconds=self.config.execution.timeout_seconds,
                working_dir=run_dir,
            )
            await self.executor.start()

            # Inject data_files into the kernel as a flat dict so generated code
            # can access paths with data_files['aff1_peaks'] without hardcoding ENCFF IDs.
            await self._inject_data_files()

            # Iteration 0: familiarize with all files (does not count toward max_iterations)
            await self._run_file_familiarization(run_dir)

            # Generate first mechanism hypothesis (no prior history)
            self.logger.info("Generating initial mechanism hypothesis")
            initial_result = await self.hypothesis_agent.refine_hypotheses(
                state=self.state,
                last_hypothesis=None,
                last_result=None,
                data_manifest=self.manifest.model_dump(),
            )

            # Store hypotheses
            for hypo in initial_result.get("hypotheses", []):
                self.state.add_hypothesis(hypo)

            if not self.state.hypotheses:
                self.logger.warning(
                    "No initial hypothesis generated — pipeline cannot proceed. "
                    "Check if the LLM returned a valid 'hypotheses' list."
                )

            # Save initial state
            self._save_state(run_dir / "state_initial.json")

            # Main loop
            tested_ids: set[int] = set()
            consecutive_hypo_rejections = 0
            max_hypo_rejections = self.config.pipeline.max_consecutive_hypothesis_rejections
            last_tested_hypothesis: dict[str, Any] | None = None
            reviewer_rejected: list[dict[str, Any]] = []  # accumulate all reviewer rejections
            consecutive_data_rejections = 0

            while (
                not self.state.converged
                and self.state.current_iteration < self.state.max_iterations
                and not self._check_shutdown()
            ):

                # Get next hypothesis to test (prefer same group as last tested)
                hypothesis = self.hypothesis_agent.get_next_hypothesis(
                    self.state.hypotheses, tested_ids, last_tested=last_tested_hypothesis
                )

                if hypothesis is None:
                    self.logger.warning("No more hypotheses to test")
                    break

                hypo_id = hypothesis.get("id", len(tested_ids))
                self.state.current_hypothesis_index = hypo_id

                # Stage 1: Review hypothesis before code generation
                # Technical retries skip review — the same hypothesis already passed once;
                # the review agent would incorrectly flag retries as duplicates
                if hypothesis.get("_is_retry"):
                    self.logger.info("Skipping hypothesis review for technical retry", {
                        "hypothesis": hypothesis.get("name", "N/A"),
                    })
                    consecutive_hypo_rejections = 0
                else:
                    hypo_review = await self.review_agent.review_hypothesis(
                        hypothesis=hypothesis,
                        tested_hypotheses=self.state.tested_hypotheses,
                    )

                    if not hypo_review.approved:
                        # If the review LLM failed, approve rather than reject —
                        # there is no static equivalent for hypothesis quality checks
                        if any("REVIEW_UNAVAILABLE" in issue for issue in hypo_review.issues_found):
                            self.logger.warning(
                                "Hypothesis review LLM unavailable, approving by default",
                                {"hypothesis": hypothesis.get("name", "N/A")}
                            )
                        else:
                            consecutive_hypo_rejections += 1

                    if not hypo_review.approved and not any(
                        "REVIEW_UNAVAILABLE" in issue for issue in hypo_review.issues_found
                    ):
                        feedback = "; ".join(hypo_review.issues_found)
                        self.logger.info("Hypothesis rejected by review", {
                            "hypothesis": hypothesis.get("name", "N/A"),
                            "feedback": feedback[:200],
                            "consecutive_rejections": consecutive_hypo_rejections,
                        })
                        self._display_message(
                            f"Hypothesis review: {feedback[:200]}",
                            "warning"
                        )

                        # IMPORTANT: Mark this hypothesis as consumed so we don't repeatedly
                        # re-review the same rejected hypothesis. Regenerated hypotheses are
                        # appended to self.state.hypotheses and will be selected next.
                        tested_ids.add(hypo_id)

                        # Record rejection so regeneration prompt knows what was already tried
                        reviewer_rejected.append({
                            "name": hypothesis.get("name", "N/A"),
                            "prediction": hypothesis.get("prediction", "N/A"),
                            "feedback": feedback,
                            "rejection_category": hypo_review.rejection_category,
                        })

                        if consecutive_hypo_rejections >= max_hypo_rejections:
                            self.logger.warning("Max consecutive hypothesis rejections reached", {
                                "count": consecutive_hypo_rejections,
                            })
                            supported = [h for h in self.state.tested_hypotheses if h.get("result") == "SUPPORTS"]
                            if supported:
                                names = ", ".join(h.get("name", "?") for h in supported)
                                self._display_message(
                                    f"Max rejections reached; synthesizing from supported evidence: {names}",
                                    "warning"
                                )
                                self.state.mark_synthesized_stop(
                                    reason="Max consecutive hypothesis rejections; synthesizing from supported evidence",
                                    confidence=0.7,
                                    conclusion=(
                                        f"Max consecutive hypothesis rejections; synthesizing from supported evidence. "
                                        f"Supported: {names}. Rejected hypotheses were duplicates or insufficiently distinct. "
                                        f"The final report should summarize the best-supported mechanism and contrast with ruled-out alternatives."
                                    ),
                                    mechanism_category_number=None,
                                    mechanism_category_name=None,
                                )
                            else:
                                self._display_message(
                                    f"Pipeline stopping: {max_hypo_rejections} consecutive hypothesis rejections",
                                    "error"
                                )
                                self.state.conclusion = (
                                    f"Pipeline stopped: hypothesis reviewer rejected "
                                    f"{max_hypo_rejections} consecutive hypotheses. "
                                    f"The hypothesis agent may be stuck generating duplicates or vague predictions."
                                )
                            break

                        # Ask hypothesis agent to regenerate
                        new_hypo = await self.hypothesis_agent.regenerate_hypothesis(
                            state=self.state,
                            rejected_hypothesis=hypothesis,
                            feedback=feedback,
                            rejection_category=hypo_review.rejection_category,
                            reviewer_rejected=reviewer_rejected,
                            data_manifest=self.manifest.model_dump(),
                        )
                        if new_hypo:
                            self.state.add_hypothesis(new_hypo)
                        else:
                            # Regeneration returned None — fall back to refine_hypotheses()
                            self.logger.warning(
                                "Regeneration returned no hypothesis, falling back to refine_hypotheses()"
                            )
                            available_bl, used_bl = self._biology_layers_for_convergence()
                            unused_bl = [k for k in available_bl if k not in used_bl] if available_bl else None
                            has_support = any(
                                h.get("result") == "SUPPORTS" and h.get("group") != "signal-authenticity"
                                for h in self.state.tested_hypotheses
                            )
                            try:
                                fallback = await self.hypothesis_agent.refine_hypotheses(
                                    state=self.state,
                                    last_hypothesis=hypothesis,
                                    last_result=self.state.tested_hypotheses[-1] if self.state.tested_hypotheses else None,
                                    data_manifest=self.manifest.model_dump(),
                                    encourage_different_mechanism=has_support,
                                    unused_biology_layers=unused_bl or None,
                                )
                                for hypo in fallback.get("hypotheses", []):
                                    self.state.add_hypothesis(hypo)
                            except Exception as e:
                                self.logger.error("Fallback hypothesis generation also failed", {"error": str(e)})
                        continue

                    # Only reset rejection counter on genuine approval (not LLM-unavailable bypass)
                    if hypo_review.approved:
                        consecutive_hypo_rejections = 0

                # Hypothesis passed review — mark as consumed
                tested_ids.add(hypo_id)

                # Stage 1.5: Deterministic data-availability check
                data_ok, missing_keys, available_keys = self._check_data_availability(hypothesis)
                if not data_ok:
                    consecutive_data_rejections += 1
                    self.logger.info("Hypothesis requires unavailable data", {
                        "hypothesis": hypothesis.get("name", "N/A"),
                        "missing_keys": missing_keys,
                        "consecutive_data_rejections": consecutive_data_rejections,
                    })
                    self._display_message(
                        f"Data pre-check: missing keys {missing_keys}",
                        "warning"
                    )
                    if consecutive_data_rejections >= 3:
                        self.logger.warning("Repeated data-availability failures, moving on")
                        self._display_message(
                            "Pipeline skipping: 3 consecutive data-availability rejections",
                            "warning"
                        )
                        consecutive_data_rejections = 0
                        continue
                    data_feedback = (
                        f"DATA_UNAVAILABLE: required_data references keys not in the manifest: "
                        f"{missing_keys}. Available keys: {available_keys}. "
                        f"You may keep the same causal mechanism if it can be tested with "
                        f"available data — just revise required_data and verification_plan."
                    )
                    new_hypo = await self.hypothesis_agent.regenerate_hypothesis(
                        state=self.state,
                        rejected_hypothesis=hypothesis,
                        feedback=data_feedback,
                        reviewer_rejected=reviewer_rejected,
                        data_manifest=self.manifest.model_dump(),
                    )
                    if new_hypo:
                        self.state.add_hypothesis(new_hypo)
                    continue
                else:
                    consecutive_data_rejections = 0

                # Only count as an iteration when a hypothesis passes review
                self.state.current_iteration += 1
                self.logger.info(f"Starting iteration {self.state.current_iteration}")

                # Request approval for hypothesis if interactive mode
                approval = await self._approve_hypothesis(
                    hypothesis, self.state.current_iteration
                )

                if approval.decision == ApprovalDecision.ABORT:
                    self._display_message("Pipeline aborted by user", "warning")
                    break
                elif approval.decision == ApprovalDecision.SKIP:
                    self._display_message(
                        f"Skipping hypothesis: {hypothesis.get('name', 'N/A')}",
                        "info"
                    )
                    continue
                elif approval.decision == ApprovalDecision.REJECT:
                    # TODO: Regenerate hypothesis with feedback
                    self._display_message(
                        f"Hypothesis rejected, feedback: {approval.feedback}",
                        "warning"
                    )
                    continue

                # Run verification cycle
                narrative.write_hypothesis(self.state.current_iteration, hypothesis)
                result = await self._run_verification_cycle(
                    hypothesis, run_dir
                )

                # Track last tested for group affinity
                last_tested_hypothesis = hypothesis

                # Update state with results
                self.state.tested_hypotheses.append({
                    **hypothesis,
                    "result": result.get("support_level", "UNKNOWN"),
                    "evidence_summary": result.get("summary", ""),
                    "review_rejections_before_success": result.get("_review_rejections", 0),
                    "executed_after_review_approval": result.get("_executed_after_review_approval", False),
                    "execution_attempt": result.get("_execution_attempt"),
                })

                self.state.add_evidence({
                    "hypothesis_id": hypo_id,
                    "hypothesis_name": hypothesis.get("name", ""),
                    "findings": result.get("findings", []),
                    "support_level": result.get("support_level", "UNKNOWN"),
                    "confidence": result.get("confidence", 0.0),
                    "summary": result.get("summary", ""),
                    "review_rejections_before_success": result.get("_review_rejections", 0),
                    "executed_after_review_approval": result.get("_executed_after_review_approval", False),
                    "execution_attempt": result.get("_execution_attempt"),
                })
                narrative.write_result(result)

                # Check for convergence or refinement
                if not self._check_shutdown():
                    # If all retries failed, stop the pipeline
                    if result.get("support_level") == "ERROR":
                        reasoning = result.get("reasoning", "Unknown error")
                        # Give an accurate stop reason instead of always saying "max retries exceeded".
                        # Static pre-gate cap and code review cap fire before any execution attempt,
                        # so "max retries exceeded" is misleading in those cases.
                        if any(
                            tag in reasoning
                            for tag in ("Static pre-gate cap", "Code review cap", "pre-gate")
                        ):
                            stop_reason = reasoning
                        else:
                            stop_reason = f"max retries ({self.config.execution.max_retries}) exceeded"
                        self.state.mark_stopped(
                            reason=stop_reason,
                            confidence=0.0,
                            conclusion=f"Pipeline stopped: {reasoning}",
                        )
                        self._display_message(
                            f"Stopping pipeline: {stop_reason}",
                            "error"
                        )
                    else:
                        await self._check_and_refine(hypothesis, result)
                        narrative.write_decision(self.state.tested_hypotheses[-1])

                # Save intermediate state after refinement so decision/reasoning are included
                if self.config.pipeline.save_intermediate:
                    self._save_state(
                        run_dir / f"state_iter_{self.state.current_iteration}.json"
                    )

            # If max iterations reached without convergence, set a default conclusion
            if not self.state.converged and not self.state.conclusion:
                supported = [h for h in self.state.tested_hypotheses if h.get("result") == "SUPPORTS"]
                refused = [h for h in self.state.tested_hypotheses if h.get("result") == "REFUSES"]
                if supported:
                    names = ", ".join(h.get("name", "unnamed") for h in supported)
                    refused_names = ", ".join(h.get("name", "unnamed") for h in refused)
                    self.state.conclusion = (
                        f"Max iterations reached. Supported mechanism evidence found "
                        f"({names}), but no mechanism achieved full convergence. "
                        f"Ruled-out mechanisms: {refused_names}."
                    )
                else:
                    self.state.conclusion = "Max iterations reached without finding a supported hypothesis."

            if (
                not self.state.converged
                and not self.state.synthesized
                and self.state.run_status == "running"
            ):
                self.state.mark_stopped(
                    reason="Max iterations reached without convergence",
                    confidence=self.state.confidence_level,
                    conclusion=self.state.conclusion or "Investigation incomplete",
                )

            # Generate final report
            narrative.finalize(
                conclusion=self.state.conclusion or "",
                converged=self.state.converged,
                total_cost=self.cost_tracker.session_cost if self.cost_tracker else None,
            )
            final_result = await self._generate_final_output(run_dir)

            return final_result

        except TokenLimitExceeded as e:
            self.logger.error("Input tokens exceed model context limit", {
                "input_tokens": e.estimate.input_tokens,
                "context_limit": e.context_limit,
            })
            self._display_message(
                f"Input too large for model! Tokens: {e.estimate.input_tokens:,}, "
                f"Limit: {e.context_limit:,}. The execution output may be too large - "
                f"consider truncating or using a model with larger context.",
                "error"
            )
            raise

        except CostLimitExceeded as e:
            self.logger.error("Per-call cost limit exceeded", {
                "estimated_cost": e.estimate.total_cost,
                "per_call_limit": e.per_call_limit,
                "input_tokens": e.estimate.input_tokens,
            })
            self._display_message(
                f"Single API call too expensive! Estimated: ${e.estimate.total_cost:.4f} "
                f"(input: {e.estimate.input_tokens} tokens), Limit: ${e.per_call_limit:.2f}",
                "error"
            )
            raise

        except SessionBudgetExceeded as e:
            self.logger.error("Session budget exceeded", {
                "estimated_cost": e.estimate.total_cost,
                "session_total": e.session_total,
                "session_limit": e.session_limit,
            })
            self._display_message(
                f"Session budget exceeded! Current: ${e.session_total:.4f}, "
                f"Next call: ${e.estimate.total_cost:.4f}, Limit: ${e.session_limit:.2f}",
                "error"
            )
            raise

        except Exception as e:
            self.logger.error("Pipeline failed", {"error": str(e)})
            raise

        finally:
            # Cleanup
            if self.executor:
                await self.executor.stop()

    async def _inject_data_files(self) -> None:
        """Inject a flat data_files dict into the kernel.

        Flattens the manifest data section so generated code can access any file as
        data_files['aff1_peaks'] instead of hardcoding ENCFF IDs or absolute paths.
        Leaf values that are strings are included directly; nested dicts are flattened
        one level with the leaf key name taking precedence over category.key notation.
        """
        data = (self.manifest.model_dump().get("data") or {})
        flat: dict[str, str] = {}
        for category, items in data.items():
            if isinstance(items, dict):
                for name, path in items.items():
                    if isinstance(path, str):
                        flat[name] = path
                        flat[f"{category}.{name}"] = path  # also expose category.name form
            elif isinstance(items, str):
                flat[category] = items

        # Serialize as a Python literal and execute in the kernel
        import json as _json
        lines = ["data_files = {"]
        for k, v in flat.items():
            lines.append(f"    {_json.dumps(k)}: {_json.dumps(v)},")
        lines.append("}")
        injection_code = "\n".join(lines)

        result = await self.executor.execute(injection_code)
        if not result.success:
            raise RuntimeError(
                f"data_files injection failed — cannot continue: "
                f"all generated code expects data_files to be defined. "
                f"stderr: {result.stderr}"
            )
        self.logger.info("Injected data_files into kernel", {"keys": list(flat.keys())})

    async def _run_file_familiarization(self, run_dir: Path) -> None:
        """Run iteration 0: inspect all manifest files and store summaries in state.

        Does not count toward max_iterations. Non-blocking — pipeline continues even if
        inspection fails.
        """
        self.logger.info("Running file familiarization (iteration 0)")
        try:
            code = await self.coding_agent.generate_inspection_code(
                self.manifest.model_dump(),
            )
            # Save code for debugging
            code_path = run_dir / "file_inspection.py"
            code_path.write_text(code)

            result = await self.executor.execute(code)
            stdout = result.stdout or ""

            if stdout.strip():
                self.state.file_summaries["all_files"] = stdout
                self.logger.info("File familiarization complete", {
                    "output_length": len(stdout),
                })
            else:
                self.logger.warning("File inspection produced no output")

        except Exception as e:
            self.logger.warning("File familiarization failed (non-fatal)", {"error": str(e)})

    async def _run_verification_cycle(
        self,
        hypothesis: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        """Run a single verification cycle for a hypothesis using the REPL agent.

        The coding agent runs an incremental execute-observe loop until it reaches
        a conclusion.  No static pre-gate or LLM code review — errors are fed back
        to the model as observations and it fixes them in place.

        Args:
            hypothesis: The hypothesis to verify
            run_dir: Directory for outputs

        Returns:
            Result summary from the verification
        """
        self.logger.info("Running verification", {
            "hypothesis": hypothesis.get("name", "N/A"),
        })

        max_repl_iterations = self.config.execution.max_retries * 4  # generous budget

        # Tell the coding agent which hypothesis iteration we're on (for debug filenames)
        self.coding_agent.state_iter = self.state.current_iteration

        result = await self.coding_agent.run_repl(
            hypothesis=hypothesis,
            executor=self.executor,
            data_manifest=self.manifest.model_dump(),
            run_dir=run_dir,
            prior_evidence=self.state.evidence,
            file_summaries=self.state.file_summaries or None,
            allowed_packages=self._allowed_packages,
            max_iterations=max_repl_iterations,
        )

        # Save combined output
        raw_output = result.get("_raw_output", "")
        output_file = run_dir / f"output_iter_{self.state.current_iteration}.txt"
        output_file.write_text(raw_output)

        self.logger.info("REPL verification complete", {
            "support_level": result.get("support_level"),
            "confidence": result.get("confidence"),
        })
        return result


    BIOLOGY_LAYERS = ("rnaseq", "phyloP", "string")  # expression / conservation / PPI; tracked for convergence nudge

    def _biology_layers_for_convergence(self) -> tuple[list[str], list[str]]:
        """Compute which biology layers are available and which were used in any tested hypothesis.

        "Used" includes any support level (SUPPORTS, REFUTES, INCONCLUSIVE) — the goal is to
        ensure the pipeline at least attempts functional/conservation characterization, not
        that it must succeed.

        Returns:
            (available_biology_layers, used_biology_layers)
        """
        data = self.manifest.model_dump().get("data", {}) or {}
        available = [k for k in self.BIOLOGY_LAYERS if k in data]
        # STRING is always available (external API, not manifest-dependent); always required
        if "string" not in available:
            available.append("string")
        used_set: set[str] = set()
        for th in self.state.tested_hypotheses:
            # QC/signal-authenticity uses rnaseq for TPM check — not functional characterization
            if th.get("group") == "signal-authenticity":
                continue
            for key in th.get("required_data", []):
                top = key.split(".", 1)[0].split(":", 1)[0]
                if top in self.BIOLOGY_LAYERS:
                    used_set.add(top)
        return available, sorted(used_set)

    def _build_group_summary(self, hypothesis: dict[str, Any]) -> str | None:
        """Build a summary if the hypothesis's group is now fully tested.

        Args:
            hypothesis: The hypothesis that was just tested

        Returns:
            Group summary string, or None if group is not complete
        """
        group = hypothesis.get("group")
        if not group:
            return None

        # Find all hypotheses in this group
        group_hypos = [h for h in self.state.hypotheses if h.get("group") == group]
        tested_ids_in_group = set()
        for th in self.state.tested_hypotheses:
            if th.get("group") == group:
                tested_ids_in_group.add(th.get("id"))

        untested_in_group = [
            h for h in group_hypos if h.get("id") not in tested_ids_in_group
        ]
        if untested_in_group:
            return None  # Group not yet complete

        # Build synthesis summary
        parts = [f'# Completed Hypothesis Group: "{group}"', ""]
        parts.append("All sub-hypotheses in this group have been tested:")
        for th in self.state.tested_hypotheses:
            if th.get("group") == group:
                parts.append(
                    f"- **{th.get('name', '?')}**: {th.get('result', '?')} — "
                    f"{th.get('evidence_summary', 'no summary')[:200]}"
                )
        parts.append("")
        parts.append(
            "Synthesize these results when deciding next steps. "
            "Consider whether the group as a whole supports, partially supports, "
            "or refuses the broad mechanism."
        )
        return "\n".join(parts)

    async def _check_and_refine(
        self,
        last_hypothesis: dict[str, Any],
        last_result: dict[str, Any],
    ) -> None:
        """Check results and potentially refine hypotheses.

        Convergence is determined first by an independent SummaryAgent check,
        then HypothesisAgent proposes the next hypothesis if not converged.

        Args:
            last_hypothesis: The hypothesis that was just tested
            last_result: Results from testing (may include '_raw_output' key)
        """
        self.logger.info("Checking results and refining")

        # Step 1: Independent convergence check — SummaryAgent uses a fresh context
        raw_output = last_result.pop("_raw_output", None)
        available_biology_layers, used_biology_layers = self._biology_layers_for_convergence()
        convergence = await self.summary_agent.check_convergence(
            finding=self.state.finding,
            tested_hypotheses=self.state.tested_hypotheses,
            last_result=last_result,
            raw_output=raw_output,
            investigation_objective=self.manifest.investigation_objective or None,
            available_biology_layers=available_biology_layers,
            used_biology_layers=used_biology_layers,
            causal_capable_data=getattr(self.manifest, "causal_capable_data", False),
        )

        has_any_supports = any(
            h.get("result") == "SUPPORTS" and h.get("group") != "signal-authenticity"
            for h in self.state.tested_hypotheses
        )

        if convergence.get("converged", False) and not has_any_supports:
            self.logger.warning(
                "Convergence check returned converged=true but no hypothesis has "
                "support_level=SUPPORTS — overriding to converged=false",
                {"tested_count": len(self.state.tested_hypotheses)},
            )
            convergence["converged"] = False
            convergence["reasoning"] = (
                f"[OVERRIDDEN] LLM declared convergence but no hypothesis achieved "
                f"SUPPORTS (all are REFUSES/INCONCLUSIVE/ERROR/UNTESTABLE). "
                f"Original reasoning: {convergence.get('reasoning', '')}"
            )

        if convergence.get("converged", False):
            if self.state.tested_hypotheses:
                self.state.tested_hypotheses[-1]["decision"] = "CONVERGED"
                self.state.tested_hypotheses[-1]["refinement_reasoning"] = convergence.get("reasoning", "")
                self.state.tested_hypotheses[-1]["mechanism_category_number"] = convergence.get("mechanism_category_number")
                self.state.tested_hypotheses[-1]["mechanism_category_name"] = convergence.get("mechanism_category_name")
            self.state.mark_converged(
                reason="Independent convergence check confirmed",
                confidence=convergence.get("confidence", 0.8),
                conclusion=convergence.get("conclusion", ""),
                mechanism_category_number=convergence.get("mechanism_category_number"),
                mechanism_category_name=convergence.get("mechanism_category_name"),
            )
            self._display_message(
                f"Converged on mechanism #{convergence.get('mechanism_category_number')}: "
                f"{convergence.get('mechanism_category_name', '')}",
                "success"
            )
            return

        # Store convergence check reasoning even when not converged
        if self.state.tested_hypotheses:
            self.state.tested_hypotheses[-1]["convergence_reasoning"] = convergence.get("reasoning", "")

        # Step 2: HypothesisAgent proposes next hypothesis
        group_summary = self._build_group_summary(last_hypothesis)
        # Only count mechanism hypotheses as SUPPORTS — QC/signal-authenticity is not a mechanism
        has_support = any(
            h.get("result") == "SUPPORTS" and h.get("group") != "signal-authenticity"
            for h in self.state.tested_hypotheses
        )
        available_bl, used_bl = self._biology_layers_for_convergence()
        unused_biology_layers = [k for k in available_bl if k not in used_bl] if available_bl else None

        refinement = await self.hypothesis_agent.refine_hypotheses(
            state=self.state,
            last_hypothesis=last_hypothesis,
            last_result=last_result,
            data_manifest=self.manifest.model_dump(),
            group_summary=group_summary,
            encourage_different_mechanism=has_support,
            unused_biology_layers=unused_biology_layers or None,
        )

        decision = refinement.get("decision", "CONTINUE")

        # Guard: HypothesisAgent no longer decides convergence — treat as NEW_HYPOTHESIS
        if decision == "CONVERGED":
            self.logger.warning(
                "HypothesisAgent returned CONVERGED — treating as NEW_HYPOTHESIS "
                "(convergence is now determined by SummaryAgent)"
            )
            decision = "NEW_HYPOTHESIS"

        # Persist decision and reasoning into the last tested hypothesis
        if self.state.tested_hypotheses:
            self.state.tested_hypotheses[-1]["decision"] = decision
            self.state.tested_hypotheses[-1]["refinement_reasoning"] = refinement.get("reasoning", "")

        if decision == "INSUFFICIENT_DATA":
            self.logger.warning(
                "INSUFFICIENT_DATA — cannot test mechanism with available data",
                {"confidence": refinement.get("confidence", 0.0)}
            )
            self._display_message("Insufficient data to test this mechanism, moving on", "warning")
            for hypo in refinement.get("hypotheses", []):
                self.state.add_hypothesis(hypo)
        elif decision == "TECHNICAL_ERROR":
            technical_issues = refinement.get("technical_issues", [])
            self.logger.warning(
                "Technical error detected - will retry with fixes",
                {"issues": technical_issues}
            )
            self._display_message(
                f"Technical issues detected: {', '.join(technical_issues[:3])}",
                "warning"
            )
            hypothesis_name = last_hypothesis.get("name", "")
            times_tested = sum(
                1 for h in self.state.tested_hypotheses
                if h.get("name", "") == hypothesis_name
            )
            if times_tested >= 2:
                self.logger.warning(
                    "Hypothesis has failed technically too many times, skipping",
                    {"hypothesis": hypothesis_name, "times_tested": times_tested}
                )
                self._display_message(
                    f"Skipping hypothesis after {times_tested} technical failures: {hypothesis_name}",
                    "warning"
                )
            else:
                retry_hypos = refinement.get("hypotheses", [])
                if not retry_hypos:
                    # LLM didn't return a hypothesis — re-queue the original
                    retry_hypos = [dict(last_hypothesis)]
                for hypo in retry_hypos:
                    hypo["_is_retry"] = True  # Skip hypothesis review for technical retries
                    hypo["_technical_issues"] = technical_issues  # Pass to coding agent
                    # Bypass name-dedup in add_hypothesis: retries intentionally
                    # re-test the same hypothesis name with a code fix
                    hypo["id"] = len(self.state.hypotheses)
                    hypo["iteration"] = self.state.current_iteration
                    self.state.hypotheses.append(hypo)
        elif decision in ("REFINE", "NEW_HYPOTHESIS"):
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
            converged=self.state.converged,
            run_status=getattr(self.state, "run_status", None),
            synthesized=getattr(self.state, "synthesized", False),
        )

        # Save report
        report_file = run_dir / "final_report.md"
        report_file.write_text(report)

        # Save final state
        self._save_state(run_dir / "state_final.json")

        self.logger.info("Pipeline complete", {
            "converged": self.state.converged,
            "run_status": getattr(self.state, "run_status", None),
            "synthesized": getattr(self.state, "synthesized", None),
            "iterations": self.state.current_iteration,
            "report_file": str(report_file),
        })

        return {
            "converged": self.state.converged,
            "convergence_reason": self.state.convergence_reason,
            "run_status": getattr(self.state, "run_status", None),
            "stop_reason": getattr(self.state, "stop_reason", None),
            "synthesized": getattr(self.state, "synthesized", None),
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
