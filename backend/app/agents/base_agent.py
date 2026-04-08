import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from app.agents.agent_state import AgentState


class BaseAgent(ABC):
    """
    Abstract base class for all KnowGen agents.

    Subclasses must implement:
        - name: class-level string identifying the agent
        - run(state): core logic that reads AgentState and returns updates
    """
    name: str = "base"

    def __init__(self):
        self.logger = logging.getLogger(f"knowgen.{self.name}")

    @abstractmethod
    def run(self, state: AgentState) -> Dict[str, Any]:
        """Execute agent logic and return state updates for LangGraph."""

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node entry-point — delegates to run with error handling."""
        self.logger.info(f"=== {self.name.upper()} AGENT RUNNING ===")
        try:
            return self.run(state)
        except Exception as e:
            self.logger.error(f"{self.name} agent failed: {e}", exc_info=True)
            raise