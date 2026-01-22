"""Main entry point for the Experiment Design Generation Pipeline."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from src.utils.config import load_config, load_data_manifest
from src.utils.logging import setup_logging, get_logger


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
        default=Path("config/config.yaml"),
        help="Path to configuration file (default: config/config.yaml)",
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

    return parser.parse_args()


async def run_pipeline(config_path: Path, manifest_path: Path, output_dir: Path | None = None) -> int:
    """Run the hypothesis generation pipeline.

    Args:
        config_path: Path to configuration file
        manifest_path: Path to data manifest file
        output_dir: Optional output directory override

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger = get_logger("main")

    try:
        # Load configuration
        logger.info("Loading configuration", {"config": str(config_path)})
        config = load_config(config_path)

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

        orchestrator = Orchestrator(config, manifest, output_dir)

        # Register shutdown handler with orchestrator
        def shutdown_handler(signum, frame):
            orchestrator.request_shutdown()

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        result = await orchestrator.run()

        logger.info("Pipeline completed", {
            "converged": result.get("converged"),
            "iterations": result.get("iterations"),
            "conclusion": result.get("conclusion", "")[:200],
        })

        # Print summary to console
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Converged: {result.get('converged')}")
        print(f"Iterations: {result.get('iterations')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"\nConclusion:\n{result.get('conclusion', 'N/A')}")
        print(f"\nFull report: {result.get('report_file')}")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error("Pipeline failed", {"error": str(e)})
        raise


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)

    logger = get_logger("main")
    logger.info("Starting Experiment Design Generation Pipeline")

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Validate inputs
    if not args.config.exists():
        logger.error("Config file not found", {"path": str(args.config)})
        return 1

    if not args.manifest.exists():
        logger.error("Data manifest not found", {"path": str(args.manifest)})
        return 1

    if args.dry_run:
        logger.info("Dry run mode - validating configuration only")
        try:
            config = load_config(args.config)
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
        return asyncio.run(run_pipeline(args.config, args.manifest, args.output))
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error("Unexpected error", {"error": str(e)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
