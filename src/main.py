"""Main entry point for ARES (Automated Regulatory Explanation System)."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from src.utils.config import load_config, load_data_manifest, flatten_manifest_data, Config
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
  python -m src.main --config config/config.yaml --manifest examples/sp1_nfya_K562/data_manifest_demo.yaml
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

    # Production ledger integration
    parser.add_argument(
        "--pair-key",
        type=str,
        default=None,
        help="Production ledger key for this run (e.g. K562_ADNP_YY1). "
             "When combined with --ledger, writes a completion row on exit.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Path to production/ledger.csv. Writes completed/failed row on exit "
             "if --pair-key is also provided.",
    )
    parser.add_argument(
        "--source-csv",
        type=str,
        default="",
        help="Passthrough field for the ledger row (records which pair-list CSV "
             "this pair came from).",
    )

    return parser.parse_args()


def _write_ledger_completion(
    ledger_path: Path,
    pair_key: str,
    status: str,
    result: dict | None,
    source_csv: str = "",
    notes: str = "",
) -> None:
    """Append a completion row to the production ledger.

    Called from both the success path (result dict available) and the failure
    paths (result=None, notes populated with the exception). Never raises —
    ledger writes must not mask the original exit condition.
    """
    try:
        from src.production.ledger import LedgerRow, append_row, make_pair_key, now_iso

        # Derive cell_line/tf_a/tf_b from pair_key for convenience querying
        parts = pair_key.split("_", 2)
        cell_line = parts[0] if len(parts) > 0 else ""
        tf_a = parts[1] if len(parts) > 1 else ""
        tf_b = parts[2] if len(parts) > 2 else ""

        row_kwargs: dict = {
            "pair_key": pair_key,
            "cell_line": cell_line,
            "tf_a": tf_a,
            "tf_b": tf_b,
            "source_csv": source_csv,
            "status": status,
            "finished_at": now_iso(),
            "notes": notes,
        }

        if result is not None:
            row_kwargs.update({
                "converged": str(result.get("converged", "")),
                "confidence": f"{result.get('confidence', 0):.2f}" if result.get("confidence") is not None else "",
                "mechanism": str(result.get("mechanism_category_name") or result.get("conclusion", "")[:120] or ""),
                "output_dir": str(result.get("output_dir") or result.get("report_file") or ""),
                "run_id": str(result.get("run_id") or ""),
                "total_cost_usd": f"{result.get('total_cost_usd', 0):.4f}" if result.get("total_cost_usd") is not None else "",
            })

        row = LedgerRow(**row_kwargs)
        append_row(ledger_path, row)
    except Exception as e:  # noqa: BLE001 — ledger failure must not mask pipeline exit
        try:
            get_logger("main").warning(
                "Failed to write ledger completion row",
                {"pair_key": pair_key, "ledger": str(ledger_path), "error": str(e)},
            )
        except Exception:
            pass


async def run_pipeline_with_config(config: Config, manifest_path: Path, output_dir: Path | None = None, pair_key: str | None = None) -> dict:
    """Run the hypothesis generation pipeline with a pre-loaded config.

    Args:
        config: Configuration object
        manifest_path: Path to data manifest file
        output_dir: Optional output directory override

    Returns:
        Result dict from orchestrator.run() augmented with output_dir and
        total_cost_usd for ledger recording. Raises on pipeline failure.
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

        orchestrator = Orchestrator(config, manifest, output_dir, manifest_path=manifest_path, pair_key=pair_key)

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

        # Augment result with fields the ledger needs.
        report_file = result.get("report_file")
        if report_file:
            result["output_dir"] = str(Path(report_file).parent)
            result["run_id"] = Path(report_file).parent.name
        try:
            if getattr(orchestrator, "cost_tracker", None) is not None:
                result["total_cost_usd"] = float(
                    getattr(orchestrator.cost_tracker, "session_cost", 0.0)
                )
        except Exception:
            pass

        return result

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

    logger.info("Starting ARES (Automated Regulatory Explanation System)")

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
        except Exception as e:
            logger.error("Configuration validation failed", {"error": str(e)})
            return 1

        # Schema validity is the weaker half of the question. `data` is typed dict[str, Any]
        # so that manifests can group files freely, which means a manifest can validate
        # cleanly while entire branches of it stay invisible to every hypothesis — nested one
        # level too deep, or holding a list where a path string belongs. Report the flattened
        # view the kernel will actually receive, and name anything that will not reach it.
        manifest_dir = Path(args.manifest).resolve().parent
        flat, dropped = flatten_manifest_data(manifest.data, manifest_dir)

        # A manifest entry is a path or a note — `SP1_peaks` sits next to `SP1_peaks_format:
        # "narrowPeak (hg38)"`, and both are injected as data_files strings. Only the former can
        # be missing from disk, so existence is checked only for values that read as paths:
        # no whitespace and a directory separator. Prose always has spaces, and the one
        # space-free non-path in the demo manifest (`directory_pattern: "{TF_NAME}_human"`) has
        # no slash either.
        def is_path_like(value: str) -> bool:
            return bool(value) and not any(c.isspace() for c in value) and "/" in value

        # Listed by qualified name. Every entry also gets a bare alias, but the bare alias is
        # what collides when two categories share a leaf name, so listing bare names hides one
        # of the pair and makes an entry that is present look absent. The qualified form is
        # one-to-one with the manifest and is what generated code reaches for in practice.
        qualified = sorted(k for k in flat if "." in k)
        shadowed_entries = {d for d, sev, _ in dropped if sev == "shadowed"}
        missing = sorted(
            (k, v) for k, v in flat.items()
            if "." in k and is_path_like(v) and not Path(v).exists()
        )

        logger.info("Configuration is valid", {
            "finding": manifest.finding,
            "data_keys": list(manifest.data.keys()),
            "causal_capable_data": manifest.causal_capable_data,
            "visible_files": len(qualified),
            "missing_paths": len(missing),
            "ignored_entries": len(dropped),
        })

        # Printed rather than logged: the console handler renders the message only, and the
        # whole point of a dry run is to read the lists.
        print()
        print(f"finding              {manifest.finding[:96]}")
        print(f"causal_capable_data  {manifest.causal_capable_data}"
              f"   ({'but-for causal test REQUIRED' if manifest.causal_capable_data else 'observational; but-for test skipped'})")
        print(f"tools                {', '.join(manifest.tools) or '(none)'}")
        print()
        missing_keys = {k for k, _ in missing}
        alias_note = (
            ', and, unless marked SHADOWED, also as data_files["<name>"]'
            if shadowed_entries
            else ', and also as data_files["<name>"]'
        )
        print(
            f"{len(qualified)} entr(ies) visible to generated code as "
            f'data_files["<category>.<name>"]{alias_note}:'
        )
        for k in qualified:
            value = flat[k]
            marks = []
            if k in missing_keys:
                marks.append("MISSING")
            if k in shadowed_entries:
                marks.append("SHADOWED")
            mark = ("  <-- " + ", ".join(marks)) if marks else ""
            shown = value if len(value) <= 66 else value[:63] + "..."
            print(f"    {k:<46} {shown}{mark}")

        if missing:
            print()
            print(f"WARNING: {len(missing)} path(s) above do not exist. Generated code will fail on them.")

        lost = [(d, r) for d, sev, r in dropped if sev == "lost"]
        shadowed = [(d, r) for d, sev, r in dropped if sev == "shadowed"]
        info = [(d, r) for d, sev, r in dropped if sev == "info"]

        if lost:
            print()
            print(f"WARNING: {len(lost)} manifest entr(ies) are unreachable — no agent will see them:")
            for dotted, reason in lost:
                print(f"    {dotted:<44} {reason}")
            print()
            print("    data_files reads data.<category>.<name> and data.<category>. Flatten the")
            print("    entries above to one of those shapes, or the pipeline ignores them.")

        if shadowed:
            print()
            print(f"{len(shadowed)} bare key(s) claimed by more than one category:")
            for dotted, reason in shadowed:
                other = reason.split('also defined by ')[1].split(';')[0]
                name = dotted.rsplit(".", 1)[1]
                print(f'    data_files["{name}"] -> {other}.{name}, shadowing {dotted}')
            print()
            print("    Not an error: every entry above is still reachable by its qualified name,")
            print("    which is what generated code uses. Rename the leaf if you want the bare")
            print("    alias to be unambiguous.")

        if info:
            print()
            print(f"{len(info)} non-path value(s) — expected; they reach the agents via the manifest,")
            print("not via data_files:")
            for dotted, reason in info:
                print(f"    {dotted:<44} {reason.split(' — ')[0]}")
        print()

        # Shadowing is deliberately not a failure: the entry is still reachable by its qualified
        # name, and that is the form generated code uses in practice. Only a path that does not
        # exist or an entry no agent can see will actually break a run.
        return 1 if (missing or lost) else 0

    # Run pipeline (with optional ledger completion hook)
    write_ledger = args.pair_key is not None and args.ledger is not None

    try:
        result = asyncio.run(run_pipeline_with_config(config, args.manifest, args.output, pair_key=args.pair_key))
        if write_ledger:
            from src.production.ledger import STATUS_COMPLETED
            _write_ledger_completion(
                args.ledger,
                args.pair_key,
                STATUS_COMPLETED,
                result,
                source_csv=args.source_csv,
            )
        return 0
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        if write_ledger:
            from src.production.ledger import STATUS_FAILED
            _write_ledger_completion(
                args.ledger,
                args.pair_key,
                STATUS_FAILED,
                None,
                source_csv=args.source_csv,
                notes="KeyboardInterrupt",
            )
        return 130
    except Exception as e:
        logger.error("Unexpected error", {"error": str(e)})
        if write_ledger:
            from src.production.ledger import STATUS_FAILED
            _write_ledger_completion(
                args.ledger,
                args.pair_key,
                STATUS_FAILED,
                None,
                source_csv=args.source_csv,
                notes=f"{type(e).__name__}: {str(e)[:400]}",
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
