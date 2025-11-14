"""
Main MMM Copilot Orchestrator

Integrates all components: response curves, optimization, RAG, visualization.
"""

from typing import Optional, Dict, List
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class MMMCopilot:
    """
    Main orchestrator for MMM Copilot.
    
    Integrates:
    - Response curve generation
    - Budget optimization
    - RAG-based query routing
    - Semantic visualization
    - Conversation context
    
    Parameters
    ----------
    data : pd.DataFrame
        MMM data with columns: date, channel, spend, impressions, predicted
    api_key : str
        OpenAI API key
    date_col : str, optional
        Name of date column (auto-detected if not provided)
    channel_col : str, optional
        Name of channel column (auto-detected if not provided)
    segment_col : str, optional
        Name of segment/taxonomy column (optional)
    agentic_mode : bool, default=False
        Enable autonomous multi-step analysis
    workspace : str, optional
        Workspace directory for artifacts
        
    Examples
    --------
    >>> copilot = MMMCopilot(data=mmm_data, api_key="sk-...")
    >>> response = copilot.query("What is TV's ROI?")
    >>> print(response['answer'])
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        api_key: str,
        *,
        date_col: Optional[str] = None,
        channel_col: Optional[str] = None,
        segment_col: Optional[str] = None,
        agentic_mode: bool = False,
        workspace: Optional[str] = None
    ):
        self.data = data
        self.api_key = api_key
        self.agentic_mode = agentic_mode
        self.workspace = workspace or "workspace"
        
        # Auto-detect columns
        date_candidates = ['date', 'week', 'month', 'day', 'week_monday', 'period']
        channel_candidates = ['channel', 'media_channel', 'medium', 'source']
        
        self.date_col = self._detect_column(date_col, date_candidates)
        self.channel_col = self._detect_column(channel_col, channel_candidates)
        self.segment_col = segment_col  # Optional
        
        # Validate data
        self._validate_data()
        
        logger.info(f"Initialized MMMCopilot with {len(data)} rows, {len(data[self.channel_col].unique())} channels")
    
    def _detect_column(self, specified: Optional[str], candidates: List[str]) -> str:
        """Auto-detect column name from candidates"""
        if specified and specified in self.data.columns:
            return specified
        
        # Try candidates
        for candidate in candidates:
            if candidate in self.data.columns:
                logger.info(f"Auto-detected column: {candidate}")
                return candidate
        
        # Not found
        raise ValueError(
            f"Could not find column. Tried: {candidates}\n"
            f"Available columns: {list(self.data.columns)}\n"
            f"Please specify explicitly using date_col= or channel_col= parameter"
        )
    
    def _validate_data(self) -> None:
        """Validate data has required columns and proper format"""
        required = [self.date_col, self.channel_col, 'spend', 'impressions', 'predicted']
        
        missing = [col for col in required if col not in self.data.columns]
        
        if missing:
            raise ValueError(
                f"Data validation failed. Missing columns: {missing}\n"
                f"Available columns: {list(self.data.columns)}\n"
                f"Required: {required}"
            )
        
        logger.info("Data validation passed")
    
    def query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Query the copilot with natural language - Uses Agentic System.
        
        Parameters
        ----------
        query : str
            Natural language query
        session_id : str, optional
            Session ID for conversation context
            
        Returns
        -------
        Dict
            Response with answer, confidence, sources, visualizations
        """
        logger.info(f"Processing query: {query}")
        
        # Use Agentic Analyzer
        from llm_copilot.core.agentic_system import AgenticAnalyzer
        import os
        
        # Allow custom OpenAI base URL via environment variable (for enterprise deployments)
        base_url = os.getenv('OPENAI_BASE_URL')
        
        analyzer = AgenticAnalyzer(
            data=self.data,
            api_key=self.api_key,
            base_url=base_url,  # None will use default OpenAI endpoint
            date_col=self.date_col
        )
        
        result = analyzer.analyze(query)
        
        # Add sources
        result['sources'] = result.get('sources', [{'type': 'agentic_analysis', 'content': 'AgenticAnalyzer'}])
        
        return result
    
    def _calculate_metrics(self) -> Dict:
        """Calculate key metrics from data"""
        metrics = {}
        
        for channel in self.data[self.channel_col].unique():
            channel_data = self.data[self.data[self.channel_col] == channel]
            
            total_spend = channel_data['spend'].sum()
            total_predicted = channel_data['predicted'].sum()
            roi = total_predicted / total_spend if total_spend > 0 else 0
            
            metrics[channel] = {
                'total_spend': total_spend,
                'total_predicted': total_predicted,
                'roi': roi,
                'avg_weekly_spend': channel_data['spend'].mean(),
                'weeks': len(channel_data)
            }
        
        return metrics
    
    def _build_context(self, metrics: Dict) -> str:
        """Build context string from metrics"""
        lines = ["Channel Performance Summary:"]
        lines.append("")
        
        # Sort by ROI descending
        sorted_channels = sorted(metrics.items(), key=lambda x: x[1]['roi'], reverse=True)
        
        for channel, data in sorted_channels:
            lines.append(f"{channel}:")
            lines.append(f"  - Total Spend: ${data['total_spend']:,.0f}")
            lines.append(f"  - Total Predicted Sales: ${data['total_predicted']:,.0f}")
            lines.append(f"  - ROI: {data['roi']:.2f}x")
            lines.append(f"  - Average Weekly Spend: ${data['avg_weekly_spend']:,.0f}")
            lines.append("")
        
        return "\n".join(lines)
    
    def analyze_autonomous(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Perform autonomous multi-step analysis.
        
        Parameters
        ----------
        query : str
            Complex analysis request
        session_id : str, optional
            Session ID
            
        Returns
        -------
        Dict
            Analysis report with findings, visualizations, recommendations
        """
        logger.info(f"Starting autonomous analysis: {query}")
        
        # Stub implementation
        return {
            'answer': 'Autonomous analysis mode requires full agentic planner integration.',
            'steps': [],
            'findings': [],
            'recommendations': []
        }
    
    def collect_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        comment: Optional[str] = None
    ) -> None:
        """
        Collect user feedback.
        
        Parameters
        ----------
        query : str
            Original query
        response : str
            Copilot response
        rating : int
            Rating 1-5
        comment : str, optional
            User comment
        """
        logger.info(f"Feedback received: rating={rating} for query='{query[:50]}'")
        # Stub - in production, would store in feedback system
