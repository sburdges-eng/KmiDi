"""Proposals and voting for orchestrator dashboard."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import AIAgent


class ProposalStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class _Proposal:
    proposal_id: str = ""
    status: str = ProposalStatus.PENDING
    proposer: Optional[AIAgent] = None
    votes_for: int = 0
    votes_against: int = 0
    title: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "status": self.status,
            "proposer": self.proposer.value if self.proposer else None,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "title": self.title,
            "summary": self.summary,
        }


class ProposalManager:
    """In-memory proposal and vote tracking for orchestrator dashboard."""

    def __init__(self) -> None:
        self._proposals: List[_Proposal] = []
        self._next_id = 0

    def _next_proposal_id(self) -> str:
        self._next_id += 1
        return f"prop_{self._next_id}"

    def submit_proposal(
        self,
        title: str = "",
        summary: str = "",
        proposer: Optional[AIAgent] = None,
        **kwargs: Any,
    ) -> Optional[_Proposal]:
        p = _Proposal(
            proposal_id=self._next_proposal_id(),
            status=ProposalStatus.PENDING,
            proposer=proposer,
            title=title or kwargs.get("title", ""),
            summary=summary or kwargs.get("summary", ""),
        )
        self._proposals.append(p)
        return p

    def vote_on_proposal(
        self,
        proposal_id: Optional[str] = None,
        approve: bool = True,
        **kwargs: Any,
    ) -> bool:
        pid = proposal_id or kwargs.get("proposal_id")
        if not pid:
            return False
        for p in self._proposals:
            if p.proposal_id == pid and p.status == ProposalStatus.PENDING:
                if approve:
                    p.votes_for += 1
                else:
                    p.votes_against += 1
                return True
        return False

    def get_all_proposals(self) -> List[_Proposal]:
        return list(self._proposals)

    def get_proposal_summary(self) -> Dict[str, Any]:
        total = len(self._proposals)
        by_status: Dict[str, int] = {}
        by_agent: Dict[str, int] = {}
        for p in self._proposals:
            by_status[p.status] = by_status.get(p.status, 0) + 1
            agent_key = p.proposer.value if p.proposer else "unknown"
            by_agent[agent_key] = by_agent.get(agent_key, 0) + 1
        return {
            "total": total,
            "by_status": ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items())),
            "by_agent": ", ".join(f"{k}: {v}" for k, v in sorted(by_agent.items())),
        }

    def get_pending_votes(self, agent: AIAgent) -> List[_Proposal]:
        return [p for p in self._proposals if p.status == ProposalStatus.PENDING]

    def get_agent_proposal_slots(self) -> Dict[AIAgent, int]:
        # Max pending proposals per agent (e.g. 5); agents with fewer get slots
        max_slots = 5
        used: Dict[AIAgent, int] = {}
        for p in self._proposals:
            if p.status == ProposalStatus.PENDING and p.proposer:
                used[p.proposer] = used.get(p.proposer, 0) + 1
        return {a: max(0, max_slots - used.get(a, 0)) for a in AIAgent}
