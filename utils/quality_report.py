from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


def generate_report(
    verified_topics: List[Dict[str, Any]],
    full_book_text: str,
    terminology_corrections: List[Dict],
) -> Dict[str, Any]:
    total_words = len(full_book_text.split())
    total_topics = len(verified_topics)

    total_verified = sum(t.get("stats", {}).get("verified", 0) for t in verified_topics)
    total_rewritten = sum(t.get("stats", {}).get("rewritten", 0) for t in verified_topics)
    total_removed = sum(t.get("stats", {}).get("removed", 0) for t in verified_topics)
    total_claims = total_verified + total_rewritten + total_removed

    pass_rate = (total_verified / total_claims * 100) if total_claims > 0 else 100.0

    topics_without_refs = sum(
        1 for t in verified_topics if not t.get("citations")
    )

    return {
        "total_topics": total_topics,
        "total_words": total_words,
        "total_claims": total_claims,
        "verified": total_verified,
        "rewritten": total_rewritten,
        "removed": total_removed,
        "verification_pass_rate": round(pass_rate, 1),
        "topics_without_refs": topics_without_refs,
        "terminology_corrections": len(terminology_corrections),
    }


def print_report(report: Dict[str, Any]) -> None:
    table = Table(title="Bookify Quality Report", show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Topics", str(report["total_topics"]))
    table.add_row("Words", f"{report['total_words']:,}")
    table.add_row("Total claims checked", str(report["total_claims"]))
    table.add_row("Verified", str(report["verified"]))
    table.add_row("Rewritten", str(report["rewritten"]))
    table.add_row("Removed", str(report["removed"]))
    table.add_row("Verification pass rate", f"{report['verification_pass_rate']}%")
    table.add_row("Topics without instructor refs", str(report["topics_without_refs"]))
    table.add_row("Terminology corrections", str(report["terminology_corrections"]))

    console.print(Panel(table, border_style="green"))
