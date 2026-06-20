
from dataclasses import dataclass, field
from typing import List, Optional

from cli.ansi import ANSI


@dataclass
class FinancialAgentConfig:
    agentRole: str
    systemPersona: str
    tools: List[str] = field(default_factory=list)
    color: str = ANSI.RESET

    # temperature: float = 0.65
