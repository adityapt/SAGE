"""
Monitoring and Observability

Tracks system metrics, API costs, errors, and performance.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import time
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Centralized metrics collector for monitoring.
    
    Tracks:
    - Query latency
    - API costs (OpenAI)
    - Error rates
    - Cache hit rates
    - User satisfaction
    
    Features:
    - Real-time metrics
    - Aggregation windows (minute, hour, day)
    - Alert thresholds
    - Persistence
    
    Examples
    --------
    >>> metrics = MetricsCollector()
    >>> metrics.record_query(query="What is TV ROI?", latency_ms=245, success=True)
    >>> metrics.record_api_cost(service="openai", cost=0.002)
    >>> stats = metrics.get_stats()
    """
    
    def __init__(
        self,
        *,
        persistence_path: Optional[Path] = None,
        alert_thresholds: Optional[Dict] = None
    ):
        self.persistence_path = persistence_path or Path("logs/metrics")
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.queries: List[Dict] = []
        self.api_costs: List[Dict] = []
        self.errors: List[Dict] = []
        self.feedback: List[Dict] = []
        
        # Aggregated metrics
        self.aggregated = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_api_cost': 0.0,
            'avg_latency_ms': 0.0,
            'avg_satisfaction': 0.0
        }
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'error_rate_pct': 5.0,  # Alert if >5% errors
            'latency_p95_ms': 2000,  # Alert if p95 latency >2s
            'cost_per_hour': 10.0,   # Alert if cost >$10/hour
            'satisfaction_below': 3.0  # Alert if avg satisfaction <3.0
        }
        
        logger.info("Initialized MetricsCollector")
    
    def record_query(
        self,
        query: str,
        latency_ms: float,
        success: bool,
        *,
        query_type: Optional[str] = None,
        channels: Optional[List[str]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Record a query execution.
        
        Parameters
        ----------
        query : str
            User query
        latency_ms : float
            Query latency in milliseconds
        success : bool
            Whether query succeeded
        query_type : str, optional
            Type of query (channel_analysis, optimization, etc.)
        channels : List[str], optional
            Channels involved
        error : str, optional
            Error message if failed
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:100],  # Truncate for storage
            'latency_ms': latency_ms,
            'success': success,
            'query_type': query_type,
            'channels': channels or [],
            'error': error
        }
        
        self.queries.append(record)
        
        # Update aggregates
        self.aggregated['total_queries'] += 1
        if success:
            self.aggregated['successful_queries'] += 1
        else:
            self.aggregated['failed_queries'] += 1
            self.errors.append(record)
        
        # Update average latency (running average)
        n = self.aggregated['total_queries']
        prev_avg = self.aggregated['avg_latency_ms']
        self.aggregated['avg_latency_ms'] = (prev_avg * (n-1) + latency_ms) / n
        
        # Check alerts
        self._check_alerts()
        
        logger.debug(f"Recorded query: success={success}, latency={latency_ms}ms")
    
    def record_api_cost(
        self,
        service: str,
        cost: float,
        *,
        operation: Optional[str] = None,
        tokens: Optional[int] = None
    ) -> None:
        """
        Record API cost.
        
        Parameters
        ----------
        service : str
            API service (openai, etc.)
        cost : float
            Cost in dollars
        operation : str, optional
            Operation type (embedding, chat, etc.)
        tokens : int, optional
            Number of tokens used
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'service': service,
            'cost': cost,
            'operation': operation,
            'tokens': tokens
        }
        
        self.api_costs.append(record)
        self.aggregated['total_api_cost'] += cost
        
        logger.debug(f"Recorded API cost: {service}={cost:.4f}")
    
    def record_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        comment: Optional[str] = None
    ) -> None:
        """
        Record user feedback.
        
        Parameters
        ----------
        query : str
            Query
        response : str
            Response
        rating : int
            Rating (1-5)
        comment : str, optional
            User comment
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:100],
            'response': response[:100],
            'rating': rating,
            'comment': comment
        }
        
        self.feedback.append(record)
        
        # Update average satisfaction
        n = len(self.feedback)
        prev_avg = self.aggregated['avg_satisfaction']
        if n == 1:
            self.aggregated['avg_satisfaction'] = rating
        else:
            self.aggregated['avg_satisfaction'] = (prev_avg * (n-1) + rating) / n
        
        logger.debug(f"Recorded feedback: rating={rating}")
    
    def get_stats(self, window: Optional[str] = None) -> Dict:
        """
        Get statistics.
        
        Parameters
        ----------
        window : str, optional
            Time window ('1h', '24h', '7d', 'all')
            
        Returns
        -------
        Dict
            Statistics
        """
        # Filter by window
        if window:
            cutoff = self._get_cutoff_time(window)
            queries = [q for q in self.queries if datetime.fromisoformat(q['timestamp']) >= cutoff]
            costs = [c for c in self.api_costs if datetime.fromisoformat(c['timestamp']) >= cutoff]
            errors = [e for e in self.errors if datetime.fromisoformat(e['timestamp']) >= cutoff]
            feedback = [f for f in self.feedback if datetime.fromisoformat(f['timestamp']) >= cutoff]
        else:
            queries = self.queries
            costs = self.api_costs
            errors = self.errors
            feedback = self.feedback
        
        if not queries:
            return self._empty_stats()
        
        # Calculate stats
        total_queries = len(queries)
        successful = sum(1 for q in queries if q['success'])
        failed = total_queries - successful
        error_rate = (failed / total_queries * 100) if total_queries > 0 else 0
        
        latencies = [q['latency_ms'] for q in queries]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95_latency = self._percentile(latencies, 95) if latencies else 0
        p99_latency = self._percentile(latencies, 99) if latencies else 0
        
        total_cost = sum(c['cost'] for c in costs)
        
        ratings = [f['rating'] for f in feedback]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            'queries': {
                'total': total_queries,
                'successful': successful,
                'failed': failed,
                'error_rate_pct': error_rate
            },
            'latency': {
                'avg_ms': avg_latency,
                'p95_ms': p95_latency,
                'p99_ms': p99_latency
            },
            'costs': {
                'total_usd': total_cost,
                'per_query_usd': total_cost / total_queries if total_queries > 0 else 0
            },
            'satisfaction': {
                'avg_rating': avg_rating,
                'num_feedback': len(feedback)
            },
            'window': window or 'all'
        }
    
    def get_error_summary(self, limit: int = 10) -> List[Dict]:
        """Get recent errors"""
        return sorted(
            self.errors,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by service"""
        breakdown = defaultdict(float)
        for cost in self.api_costs:
            breakdown[cost['service']] += cost['cost']
        return dict(breakdown)
    
    def get_query_type_distribution(self) -> Dict[str, int]:
        """Get distribution of query types"""
        distribution = defaultdict(int)
        for query in self.queries:
            qtype = query.get('query_type', 'unknown')
            distribution[qtype] += 1
        return dict(distribution)
    
    def export_metrics(self, filepath: Optional[Path] = None) -> None:
        """Export metrics to JSON file"""
        filepath = filepath or (self.persistence_path / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        data = {
            'exported_at': datetime.now().isoformat(),
            'aggregated': self.aggregated,
            'stats': self.get_stats(),
            'queries': self.queries,
            'api_costs': self.api_costs,
            'errors': self.errors,
            'feedback': self.feedback
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to {filepath}")
    
    def _check_alerts(self) -> None:
        """Check alert thresholds and log warnings"""
        stats = self.get_stats('1h')
        
        # Error rate alert
        if stats['queries']['error_rate_pct'] > self.alert_thresholds['error_rate_pct']:
            logger.warning(
                f"ALERT: High error rate: {stats['queries']['error_rate_pct']:.1f}% "
                f"(threshold: {self.alert_thresholds['error_rate_pct']:.1f}%)"
            )
        
        # Latency alert
        if stats['latency']['p95_ms'] > self.alert_thresholds['latency_p95_ms']:
            logger.warning(
                f"ALERT: High latency: P95={stats['latency']['p95_ms']:.0f}ms "
                f"(threshold: {self.alert_thresholds['latency_p95_ms']}ms)"
            )
        
        # Cost alert
        if stats['costs']['total_usd'] > self.alert_thresholds['cost_per_hour']:
            logger.warning(
                f"ALERT: High cost: ${stats['costs']['total_usd']:.2f}/hour "
                f"(threshold: ${self.alert_thresholds['cost_per_hour']:.2f}/hour)"
            )
        
        # Satisfaction alert
        if stats['satisfaction']['avg_rating'] < self.alert_thresholds['satisfaction_below']:
            logger.warning(
                f"ALERT: Low satisfaction: {stats['satisfaction']['avg_rating']:.1f}/5 "
                f"(threshold: {self.alert_thresholds['satisfaction_below']:.1f}/5)"
            )
    
    def _get_cutoff_time(self, window: str) -> datetime:
        """Get cutoff time for window"""
        now = datetime.now()
        
        if window == '1h':
            return now - timedelta(hours=1)
        elif window == '24h':
            return now - timedelta(hours=24)
        elif window == '7d':
            return now - timedelta(days=7)
        else:
            return datetime.min
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _empty_stats(self) -> Dict:
        """Return empty stats structure"""
        return {
            'queries': {'total': 0, 'successful': 0, 'failed': 0, 'error_rate_pct': 0},
            'latency': {'avg_ms': 0, 'p95_ms': 0, 'p99_ms': 0},
            'costs': {'total_usd': 0, 'per_query_usd': 0},
            'satisfaction': {'avg_rating': 0, 'num_feedback': 0},
            'window': 'all'
        }


# Global metrics instance
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

