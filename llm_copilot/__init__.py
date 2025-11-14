"""
LLM Copilot for Marketing Mix Modeling

Production-ready copilot integrating response curves, optimization, and RAG-based querying.
"""

from llm_copilot.core.response_curves import ResponseCurveGenerator
from llm_copilot.core.optimization import BudgetOptimizer, OptimizationResult
from llm_copilot.core.knowledge_base import KnowledgeBase
from llm_copilot.core.copilot import MMMCopilot

__version__ = "1.0.0"

__all__ = [
    "ResponseCurveGenerator",
    "BudgetOptimizer",
    "OptimizationResult",
    "KnowledgeBase",
    "MMMCopilot",
]

