"""
RAG-based Query Router

Routes user queries to appropriate handlers based on semantic understanding.
Determines whether query needs response curve generation, optimization, or descriptive answer.
"""

from typing import Dict, Literal, Optional
from enum import Enum
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries the copilot can handle"""
    DESCRIPTIVE = "descriptive"  # What is X's ROI?
    RESPONSE_CURVE = "response_curve"  # Show me the response curve for X
    OPTIMIZATION = "optimization"  # What's the optimal allocation?
    COMPARISON = "comparison"  # Compare X vs Y
    OUT_OF_SCOPE = "out_of_scope"  # Cannot answer


class RAGRouter:
    """
    Route queries to appropriate handlers using semantic classification.
    
    Uses LLM-based classification to determine query intent and route
    to the correct handler (descriptive QA, curve generation, or optimization).
    
    Parameters
    ----------
    api_key : str
        OpenAI API key
    model : str, default="gpt-4o"
        LLM model for classification
    temperature : float, default=0.1
        Temperature for classification (low for consistent routing)
        
    Examples
    --------
    >>> router = RAGRouter(api_key="...")
    >>> result = router.route_query("What is TV's ROI?")
    >>> print(result['query_type'])  # 'descriptive'
    >>> 
    >>> result = router.route_query("What's the optimal budget allocation?")
    >>> print(result['query_type'])  # 'optimization'
    """
    
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o",
        temperature: float = 0.1
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
        logger.info(f"Initialized RAGRouter with model={model}")
    
    def route_query(self, query: str) -> Dict[str, str]:
        """
        Route query to appropriate handler.
        
        Parameters
        ----------
        query : str
            User query in natural language
            
        Returns
        -------
        Dict[str, str]
            Dictionary with 'query_type', 'rationale', and routing information
        """
        logger.info(f"Routing query: {query[:100]}...")
        
        # Use LLM-based classification (no keyword shortcuts)
        return self._llm_classify(query)
    
    def _llm_classify(self, query: str) -> Dict[str, str]:
        """
        Use LLM to classify query when keywords don't match.
        
        Parameters
        ----------
        query : str
            User query
            
        Returns
        -------
        Dict[str, str]
            Classification result
        """
        system_prompt = """You are a query classifier for a Marketing Mix Modeling copilot.

Classify queries into one of these types:
1. DESCRIPTIVE - Questions about specific metrics (ROI, spend, saturation, etc.)
2. RESPONSE_CURVE - Requests for curve visualization or saturation analysis
3. OPTIMIZATION - Requests for optimal budget allocation
4. COMPARISON - Comparing multiple channels
5. OUT_OF_SCOPE - Cannot be answered (e.g., counterfactual "what if" scenarios without simulation)

Respond with ONLY the classification type in uppercase, followed by a colon and brief rationale.

Examples:
"What is TV's ROI?" → DESCRIPTIVE: Asking for specific metric
"Show me response curves for all channels" → RESPONSE_CURVE: Requesting curve visualization
"What's the optimal budget allocation?" → OPTIMIZATION: Requires optimization algorithm
"Compare TV vs Search" → COMPARISON: Comparing channels
"What if we had spent 50% more last year?" → OUT_OF_SCOPE: Requires counterfactual simulation"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=self.temperature,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse response
            if ':' in result:
                query_type_str, rationale = result.split(':', 1)
                query_type_str = query_type_str.strip().lower()
                rationale = rationale.strip()
            else:
                query_type_str = result.lower()
                rationale = "LLM classification"
            
            # Map to QueryType
            if 'descriptive' in query_type_str:
                query_type = QueryType.DESCRIPTIVE
            elif 'response' in query_type_str or 'curve' in query_type_str:
                query_type = QueryType.RESPONSE_CURVE
            elif 'optimization' in query_type_str or 'optimal' in query_type_str:
                query_type = QueryType.OPTIMIZATION
            elif 'comparison' in query_type_str or 'compare' in query_type_str:
                query_type = QueryType.COMPARISON
            else:
                query_type = QueryType.OUT_OF_SCOPE
            
            logger.info(f"LLM classified as: {query_type}")
            
            return {
                'query_type': query_type,
                'rationale': rationale,
                'classification_method': 'llm'
            }
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Default to descriptive on failure
            return {
                'query_type': QueryType.DESCRIPTIVE,
                'rationale': f'Classification failed, defaulting to descriptive: {e}',
                'classification_method': 'fallback'
            }
    
    def requires_curves(self, query_type: QueryType) -> bool:
        """Check if query type requires response curves"""
        return query_type in [QueryType.RESPONSE_CURVE, QueryType.OPTIMIZATION]
    
    def requires_optimization(self, query_type: QueryType) -> bool:
        """Check if query type requires optimization"""
        return query_type == QueryType.OPTIMIZATION

