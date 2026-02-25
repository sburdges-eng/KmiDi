"""
Proposal formatting helpers for CLI output.
"""

from typing import List, Dict, Any


def format_proposal(proposal: Dict[str, Any]) -> str:
    title = proposal.get("title", "Untitled")
    status = proposal.get("status", "pending")
    return f"{title} [{status}]"


def format_proposal_list(proposals: List[Dict[str, Any]]) -> str:
    if not proposals:
        return "No proposals available."
    return "\n".join(format_proposal(p) for p in proposals)


__all__ = ["format_proposal", "format_proposal_list"]
