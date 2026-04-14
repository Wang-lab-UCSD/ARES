"""Main entry point for ARES (Automated Regulatory Solver)."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from src.utils.config import load_config, load_data_manifest, Config
from src.utils.logging import setup_logging, get_logger
from src.utils.model_selector import interactive_model_selection, selections_to_config


# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    logger = get_logger("main")
    logger.warning("Shutdown requested (Ctrl+C). Will exit after current step...")
    _shutdown_requested = True


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated hypothesis generation and verification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --config config/config.yaml --manifest examples/atf6_rest/data_manifest.yaml
  python -m src.main -c config.yaml -m data.yaml --verbose
        """,
    )

    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to configuration file. If not provided, interactive model selection is used.",
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Force interactive model selection even if config file exists",
    )

    parser.add_argument(
        "-m", "--manifest",
        type=Path,
        required=True,
        help="Path to data manifest file describing available data",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (overrides config)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running pipeline",
    )

    # Cost control arguments
    parser.add_argument(
        "--per-call-limit",
        type=float,
        default=None,
        metavar="USD",
        help="Maximum cost for a single API call in USD (default: 10.0, prevents runaway token usage)",
    )

    parser.add_argument(
        "--session-limit",
        type=float,
        default=None,
        metavar="USD",
        help="Maximum total cost for the session in USD (default: 50.0)",
    )

    parser.add_argument(
        "--no-cost-limit",
        action="store_true",
        help="Disable cost blocking (not recommended)",
    )

    # Interactive approval mode
    parser.add_argument(
        "-I", "--interactive-approval",
        action="store_true",
        help="Enable interactive approval mode for hypotheses and code",
    )

    return parser.parse_args()


async def run_pipeline_with_config(config: Config, manifest_path: Path, output_dir: Path | None = None) -> int:
    """Run the hypothesis generation pipeline with a pre-loaded config.

    Args:
        config: Configuration object
        manifest_path: Path to data manifest file
        output_dir: Optional output directory override

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger = get_logger("main")

    try:
        # Load data manifest
        logger.info("Loading data manifest", {"manifest": str(manifest_path)})
        manifest = load_data_manifest(manifest_path)

        logger.info("Pipeline configured", {
            "finding": manifest.finding[:100] + "..." if len(manifest.finding) > 100 else manifest.finding,
            "max_iterations": config.pipeline.max_iterations,
            "hypothesis_model": f"{config.llm.hypothesis_model.provider}/{config.llm.hypothesis_model.model}",
            "coding_model": f"{config.llm.coding_model.provider}/{config.llm.coding_model.model}",
        })

        # Initialize and run orchestrator
        from src.orchestrator import Orchestrator

        orchestrator = Orchestrator(config, manifest, output_dir, manifest_path=manifest_path)

        # Register shutdown handler with orchestrator
        def shutdown_handler(signum, frame):
            orchestrator.request_shutdown()

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        result = await orchestrator.run()

        logger.info("Pipeline completed", {
            "converged": result.get("converged"),
            "run_status": result.get("run_status"),
            "synthesized": result.get("synthesized"),
            "iterations": result.get("iterations"),
            "conclusion": (result.get("conclusion") or "")[:200],
        })

        # Print summary to console
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Run status: {result.get('run_status')}")
        print(f"Converged (strict): {result.get('converged')}")
        if result.get("run_status") and result.get("run_status") != "converged":
            print(f"Stop reason: {result.get('stop_reason') or 'N/A'}")
        print(f"Iterations: {result.get('iterations')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"\nConclusion:\n{result.get('conclusion') or 'N/A'}")
        print(f"\nFull report: {result.get('report_file')}")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error("Pipeline failed", {"error": str(e)})
        raise


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Peek at the config file for log_dir BEFORE setting up logging so
    # all entries land in the configured directory (e.g. logs_3) rather
    # than the default logs/.
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    bootstrap_log_dir = "logs"
    if args.config is not None and args.config.exists():
        try:
            import yaml
            with open(args.config) as _cf:
                _cfg_raw = yaml.safe_load(_cf) or {}
            bootstrap_log_dir = (
                (_cfg_raw.get("pipeline") or {}).get("log_dir") or "logs"
            )
        except Exception:
            pass  # fall back to default on any parse error
    setup_logging(level=level, log_dir=bootstrap_log_dir)

    logger = get_logger("main")

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Validate manifest
    if not args.manifest.exists():
        logger.error("Data manifest not found", {"path": str(args.manifest)})
        return 1

    # Determine if we should use interactive model selection
    use_interactive = args.interactive or args.config is None

    if not use_interactive and args.config is not None and not args.config.exists():
        logger.warning("Config file not found, switching to interactive mode", {"path": str(args.config)})
        use_interactive = True

    # Load or create configuration
    if use_interactive:
        logger.info("Starting interactive model selection")
        try:
            selections = interactive_model_selection()
            config_dict = selections_to_config(selections)
            config = Config(**config_dict)
        except SystemExit:
            return 1
        except Exception as e:
            logger.error("Model selection failed", {"error": str(e)})
            return 1
    else:
        try:
            config = load_config(args.config)
        except Exception as e:
            logger.error("Failed to load config", {"error": str(e)})
            return 1

    logger.info("Starting ARES (Automated Regulatory Solver)")

    # Apply CLI overrides for cost tracking
    if args.no_cost_limit:
        config.cost.enabled = False
        logger.warning("Cost blocking disabled via --no-cost-limit")
    else:
        if args.per_call_limit is not None:
            config.cost.per_call_limit_usd = args.per_call_limit
            config.cost.enabled = True
            logger.info(f"Per-call cost limit set to ${args.per_call_limit:.2f}")
        if args.session_limit is not None:
            config.cost.session_limit_usd = args.session_limit
            config.cost.enabled = True
            logger.info(f"Session cost limit set to ${args.session_limit:.2f}")

    # Apply CLI override for interactive approval
    if args.interactive_approval:
        config.interactive.enabled = True
        logger.info("Interactive approval mode enabled")

    # Dry run mode
    if args.dry_run:
        logger.info("Dry run mode - validating configuration only")
        try:
            manifest = load_data_manifest(args.manifest)
            logger.info("Configuration is valid", {
                "finding": manifest.finding,
                "data_keys": list(manifest.data.keys()),
            })
            return 0
        except Exception as e:
            logger.error("Configuration validation failed", {"error": str(e)})
            return 1

    # Run pipeline
    try:
        return asyncio.run(run_pipeline_with_config(config, args.manifest, args.output))
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error("Unexpected error", {"error": str(e)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
