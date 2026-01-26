"""Interactive approval UI for hypotheses and code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ApprovalDecision(Enum):
    """User decision for approval requests."""

    APPROVE = "approve"
    REJECT = "reject"
    SKIP = "skip"  # Skip this item, move to next
    ABORT = "abort"  # Stop the entire pipeline


@dataclass
class ApprovalResult:
    """Result of an approval request."""

    decision: ApprovalDecision
    feedback: str | None = None  # User feedback for rejection


class ApprovalUI:
    """Terminal UI for interactive approval.

    Uses rich library for enhanced display if available,
    falls back to plain text otherwise.
    """

    def __init__(self, use_rich: bool = True):
        """Initialize approval UI.

        Args:
            use_rich: Whether to use rich library for enhanced display
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        if self.use_rich:
            self.console = Console()

    def display_hypothesis(self, hypothesis: dict[str, Any], iteration: int = 0) -> None:
        """Display a hypothesis for review."""
        if self.use_rich:
            self._display_hypothesis_rich(hypothesis, iteration)
        else:
            self._display_hypothesis_plain(hypothesis, iteration)

    def _display_hypothesis_rich(self, hypothesis: dict[str, Any], iteration: int) -> None:
        """Display hypothesis using rich formatting."""
        table = Table(title=f"Hypothesis (Iteration {iteration})", show_header=False, box=None)
        table.add_column("Field", style="cyan", width=15)
        table.add_column("Value", overflow="fold")

        table.add_row("Name", hypothesis.get("name", "N/A"))

        if hypothesis.get("description"):
            table.add_row("Description", hypothesis["description"])

        if hypothesis.get("rationale"):
            table.add_row("Rationale", hypothesis["rationale"])

        if hypothesis.get("prediction"):
            table.add_row("Prediction", hypothesis["prediction"])

        if hypothesis.get("verification_plan"):
            plan = hypothesis["verification_plan"]
            if isinstance(plan, list):
                plan = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
            table.add_row("Verification", plan)

        table.add_row("Priority", str(hypothesis.get("priority", "N/A")))

        self.console.print()
        self.console.print(Panel(table, title="[bold blue]Hypothesis Review[/bold blue]"))

    def _display_hypothesis_plain(self, hypothesis: dict[str, Any], iteration: int) -> None:
        """Display hypothesis using plain text."""
        print(f"\n{'='*70}")
        print(f"HYPOTHESIS REVIEW (Iteration {iteration})")
        print(f"{'='*70}")
        print(f"Name: {hypothesis.get('name', 'N/A')}")

        if hypothesis.get("description"):
            print(f"\nDescription:\n  {hypothesis['description']}")

        if hypothesis.get("rationale"):
            print(f"\nRationale:\n  {hypothesis['rationale']}")

        if hypothesis.get("prediction"):
            print(f"\nPrediction:\n  {hypothesis['prediction']}")

        if hypothesis.get("verification_plan"):
            plan = hypothesis["verification_plan"]
            print("\nVerification Plan:")
            if isinstance(plan, list):
                for i, step in enumerate(plan, 1):
                    print(f"  {i}. {step}")
            else:
                print(f"  {plan}")

        print(f"\nPriority: {hypothesis.get('priority', 'N/A')}")
        print(f"{'='*70}\n")

    def display_code(self, code: str, hypothesis_name: str = "") -> None:
        """Display generated code for review."""
        if self.use_rich:
            title = f"Generated Code for: {hypothesis_name}" if hypothesis_name else "Generated Code"
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            self.console.print()
            self.console.print(Panel(syntax, title=f"[bold green]{title}[/bold green]"))
        else:
            print(f"\n{'='*70}")
            title = f"GENERATED CODE{' for: ' + hypothesis_name if hypothesis_name else ''}"
            print(title)
            print(f"{'='*70}")
            for i, line in enumerate(code.split('\n'), 1):
                print(f"{i:4d} | {line}")
            print(f"{'='*70}\n")

    def display_cost_summary(self, cost_summary: dict[str, Any]) -> None:
        """Display current cost tracking summary."""
        if self.use_rich:
            table = Table(title="Cost Summary", show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Session Cost", f"${cost_summary['session_cost']:.4f}")
            table.add_row("Per-Call Limit", f"${cost_summary['per_call_limit']:.2f}")
            table.add_row("Session Limit", f"${cost_summary['session_limit']:.2f}")
            table.add_row("Remaining", f"${cost_summary['remaining_budget']:.4f}")
            table.add_row("API Calls", str(cost_summary['call_count']))

            self.console.print()
            if cost_summary.get('approaching_limit'):
                self.console.print("[bold red]WARNING: Approaching session limit![/bold red]")
            self.console.print(table)
        else:
            print(f"\n--- Cost Summary ---")
            print(f"Session Cost:   ${cost_summary['session_cost']:.4f}")
            print(f"Per-Call Limit: ${cost_summary['per_call_limit']:.2f}")
            print(f"Session Limit:  ${cost_summary['session_limit']:.2f}")
            print(f"Remaining:      ${cost_summary['remaining_budget']:.4f}")
            print(f"API Calls:      {cost_summary['call_count']}")
            if cost_summary.get('approaching_limit'):
                print("WARNING: Approaching session limit!")
            print("---\n")

    def request_approval(self, item_type: str) -> ApprovalResult:
        """Request user approval for an item.

        Args:
            item_type: Description of what is being approved (e.g., "hypothesis", "code")

        Returns:
            ApprovalResult with decision and optional feedback
        """
        if self.use_rich:
            return self._request_approval_rich(item_type)
        else:
            return self._request_approval_plain(item_type)

    def _request_approval_rich(self, item_type: str) -> ApprovalResult:
        """Request approval using rich prompts."""
        self.console.print()
        self.console.print(f"[bold]Approve this {item_type}?[/bold]")
        self.console.print("  [green](a)pprove[/green] - Continue with this item")
        self.console.print("  [red](r)eject[/red]  - Provide feedback for regeneration")
        self.console.print("  [yellow](s)kip[/yellow]    - Skip this item")
        self.console.print("  [magenta](q)uit[/magenta]    - Abort pipeline")

        while True:
            choice = Prompt.ask("Choice", choices=["a", "r", "s", "q"], default="a")

            if choice == "a":
                return ApprovalResult(ApprovalDecision.APPROVE)
            elif choice == "r":
                feedback = Prompt.ask("Enter feedback for regeneration")
                return ApprovalResult(ApprovalDecision.REJECT, feedback=feedback)
            elif choice == "s":
                return ApprovalResult(ApprovalDecision.SKIP)
            elif choice == "q":
                if Confirm.ask("Are you sure you want to abort?"):
                    return ApprovalResult(ApprovalDecision.ABORT)

    def _request_approval_plain(self, item_type: str) -> ApprovalResult:
        """Request approval using plain input."""
        print(f"\nApprove this {item_type}?")
        print("  (a)pprove - Continue with this item")
        print("  (r)eject  - Provide feedback for regeneration")
        print("  (s)kip    - Skip this item")
        print("  (q)uit    - Abort pipeline")

        while True:
            try:
                choice = input("Choice [a/r/s/q]: ").strip().lower()
            except EOFError:
                # Handle non-interactive mode
                print("Non-interactive mode detected, auto-approving")
                return ApprovalResult(ApprovalDecision.APPROVE)

            if choice == "a" or choice == "":
                return ApprovalResult(ApprovalDecision.APPROVE)
            elif choice == "r":
                feedback = input("Enter feedback for regeneration: ").strip()
                return ApprovalResult(ApprovalDecision.REJECT, feedback=feedback)
            elif choice == "s":
                return ApprovalResult(ApprovalDecision.SKIP)
            elif choice == "q":
                confirm = input("Are you sure you want to abort? [y/N]: ").strip().lower()
                if confirm == "y":
                    return ApprovalResult(ApprovalDecision.ABORT)
            else:
                print("Invalid choice. Please enter a, r, s, or q.")

    def display_message(self, message: str, style: str = "info") -> None:
        """Display a message to the user.

        Args:
            message: The message to display
            style: Style hint - "info", "warning", "error", "success"
        """
        if self.use_rich:
            style_map = {
                "info": "blue",
                "warning": "yellow",
                "error": "red",
                "success": "green",
            }
            color = style_map.get(style, "white")
            self.console.print(f"[{color}]{message}[/{color}]")
        else:
            prefix_map = {
                "info": "[INFO]",
                "warning": "[WARNING]",
                "error": "[ERROR]",
                "success": "[SUCCESS]",
            }
            prefix = prefix_map.get(style, "")
            print(f"{prefix} {message}")

    def display_hypotheses_summary(
        self,
        hypotheses: list[dict[str, Any]],
        title: str = "Generated Hypotheses",
    ) -> None:
        """Display a summary of multiple hypotheses."""
        if self.use_rich:
            table = Table(title=title)
            table.add_column("#", style="dim", width=3)
            table.add_column("Priority", width=8)
            table.add_column("Name", overflow="fold")

            for i, hypo in enumerate(hypotheses, 1):
                table.add_row(
                    str(i),
                    str(hypo.get("priority", "-")),
                    hypo.get("name", "Unnamed"),
                )

            self.console.print()
            self.console.print(table)
        else:
            print(f"\n--- {title} ---")
            for i, hypo in enumerate(hypotheses, 1):
                print(f"  {i}. [P{hypo.get('priority', '?')}] {hypo.get('name', 'Unnamed')}")
            print("---\n")
