"""
User Feedback and Human-in-the-Loop

Implements user-centric feedback loops and interactive refinement.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """Single feedback entry"""
    timestamp: str
    query: str
    copilot_answer: str
    user_feedback: str  # 'correct', 'incorrect', 'partially_correct'
    user_correction: Optional[str] = None
    user_rating: Optional[int] = None  # 1-5 stars
    context: Dict = field(default_factory=dict)
    reasoning_shown: bool = False
    

class UserFeedbackManager:
    """
    Manage user feedback to improve copilot over time.
    
    Features:
    - Collect feedback on answers
    - Learn from corrections
    - Track accuracy metrics
    - Identify problematic patterns
    - Generate improvement suggestions
    
    Examples
    --------
    >>> feedback_mgr = UserFeedbackManager(workspace="./feedback/")
    >>> feedback_mgr.record_feedback(
    ...     query="What is TV's ROI?",
    ...     answer="TV ROI is 1.32x",
    ...     feedback="incorrect",
    ...     correction="TV ROI is 1.45x not 1.32x"
    ... )
    >>> accuracy = feedback_mgr.get_accuracy_stats()
    """
    
    def __init__(self, workspace: str = "./feedback/"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.workspace / "feedback_log.jsonl"
        self.corrections_file = self.workspace / "corrections.json"
        
        # Load existing corrections
        self.corrections = self._load_corrections()
        
        logger.info(f"Initialized FeedbackManager at {workspace}")
    
    def record_feedback(
        self,
        query: str,
        answer: str,
        feedback: str,
        *,
        correction: Optional[str] = None,
        rating: Optional[int] = None,
        context: Optional[Dict] = None,
        reasoning: Optional[str] = None
    ) -> None:
        """
        Record user feedback on an answer.
        
        Parameters
        ----------
        query : str
            Original question
        answer : str
            Copilot's answer
        feedback : str
            User feedback: 'correct', 'incorrect', 'partially_correct'
        correction : str, optional
            User's correction if answer was wrong
        rating : int, optional
            User rating 1-5 stars
        context : Dict, optional
            Additional context
        reasoning : str, optional
            Chain-of-thought reasoning if shown
        """
        entry = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            query=query,
            copilot_answer=answer,
            user_feedback=feedback,
            user_correction=correction,
            user_rating=rating,
            context=context or {},
            reasoning_shown=reasoning is not None
        )
        
        # Append to log
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(entry.__dict__) + '\n')
        
        # If correction provided, store for future reference
        if correction and feedback == 'incorrect':
            self._store_correction(query, answer, correction)
        
        logger.info(f"Recorded feedback: {feedback} (rating: {rating})")
    
    def _store_correction(self, query: str, wrong_answer: str, correction: str) -> None:
        """Store correction for learning"""
        correction_key = query.lower()[:100]  # Use query prefix as key
        
        if correction_key not in self.corrections:
            self.corrections[correction_key] = []
        
        self.corrections[correction_key].append({
            'query': query,
            'wrong_answer': wrong_answer,
            'correction': correction,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save to file
        with open(self.corrections_file, 'w') as f:
            json.dumps(self.corrections, f, indent=2)
    
    def _load_corrections(self) -> Dict:
        """Load stored corrections"""
        if self.corrections_file.exists():
            with open(self.corrections_file) as f:
                return json.load(f)
        return {}
    
    def get_accuracy_stats(self) -> Dict:
        """
        Get accuracy statistics from feedback.
        
        Returns
        -------
        Dict
            {
                'total': int,
                'correct': int,
                'incorrect': int,
                'accuracy': float,
                'avg_rating': float
            }
        """
        if not self.feedback_file.exists():
            return {'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0.0}
        
        feedback_entries = []
        with open(self.feedback_file) as f:
            for line in f:
                feedback_entries.append(json.loads(line))
        
        total = len(feedback_entries)
        correct = sum(1 for e in feedback_entries if e['user_feedback'] == 'correct')
        incorrect = sum(1 for e in feedback_entries if e['user_feedback'] == 'incorrect')
        
        ratings = [e['user_rating'] for e in feedback_entries if e.get('user_rating')]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        return {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'partially_correct': total - correct - incorrect,
            'accuracy': correct / total if total > 0 else 0.0,
            'avg_rating': avg_rating
        }
    
    def get_problematic_queries(self, threshold: int = 2) -> List[Dict]:
        """
        Identify queries with repeated incorrect answers.
        
        Parameters
        ----------
        threshold : int, default=2
            Minimum number of incorrect attempts
            
        Returns
        -------
        List[Dict]
            Problematic query patterns
        """
        if not self.feedback_file.exists():
            return []
        
        query_errors = {}
        with open(self.feedback_file) as f:
            for line in f:
                entry = json.loads(line)
                if entry['user_feedback'] == 'incorrect':
                    query_key = entry['query'].lower()[:100]
                    if query_key not in query_errors:
                        query_errors[query_key] = []
                    query_errors[query_key].append(entry)
        
        # Filter by threshold
        problematic = [
            {'query': k, 'errors': v, 'count': len(v)}
            for k, v in query_errors.items()
            if len(v) >= threshold
        ]
        
        return sorted(problematic, key=lambda x: x['count'], reverse=True)
    
    def suggest_improvements(self) -> List[str]:
        """Generate improvement suggestions based on feedback"""
        suggestions = []
        
        stats = self.get_accuracy_stats()
        problematic = self.get_problematic_queries()
        
        if stats['accuracy'] < 0.8:
            suggestions.append(
                f"Overall accuracy {stats['accuracy']:.1%} is below 80%. "
                "Consider adding more few-shot examples or improving prompts."
            )
        
        if len(problematic) > 0:
            suggestions.append(
                f"Found {len(problematic)} query patterns with repeated errors. "
                f"Top issue: '{problematic[0]['query']}' ({problematic[0]['count']} errors). "
                "Consider adding specific few-shot examples for these patterns."
            )
        
        if stats['avg_rating'] < 3.5:
            suggestions.append(
                f"Average rating {stats['avg_rating']:.1f}/5.0 is below 3.5. "
                "Users may find answers incomplete or unhelpful. "
                "Consider more detailed explanations or validation suggestions."
            )
        
        return suggestions


class HumanInTheLoop:
    """
    Interactive refinement with human oversight.
    
    Features:
    - Pause for human review before critical actions
    - Allow human to edit/approve recommendations
    - Interactive clarification of ambiguous queries
    - Collaborative optimization
    
    Examples
    --------
    >>> hitl = HumanInTheLoop()
    >>> # Copilot proposes optimization
    >>> proposal = {"TV": 420000, "Search": 380000}
    >>> approved = hitl.request_approval(
    ...     action="Budget reallocation",
    ...     proposal=proposal,
    ...     reasoning="Maximizes ROI..."
    ... )
    >>> if approved:
    ...     execute_optimization(proposal)
    """
    
    def __init__(self, auto_approve: bool = False):
        """
        Parameters
        ----------
        auto_approve : bool, default=False
            If True, auto-approve all actions (for testing)
        """
        self.auto_approve = auto_approve
        self.approval_history: List[Dict] = []
    
    def request_approval(
        self,
        action: str,
        proposal: Dict,
        reasoning: str,
        *,
        risk_level: str = 'medium'
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Request human approval for an action.
        
        Parameters
        ----------
        action : str
            Description of action (e.g., "Budget reallocation")
        proposal : Dict
            Proposed changes
        reasoning : str
            Why this action is recommended
        risk_level : str, default='medium'
            'low', 'medium', 'high' - determines approval requirement
            
        Returns
        -------
        Tuple[bool, Optional[Dict]]
            (approved, modified_proposal)
        """
        logger.info(f"Requesting approval for: {action}")
        
        if self.auto_approve and risk_level == 'low':
            logger.info("Auto-approved (low risk)")
            return True, proposal
        
        # Log approval request
        logger.info(f"Approval required: {action} (Risk: {risk_level})")
        logger.debug(f"Reasoning: {reasoning}")
        logger.debug(f"Proposal: {proposal}")
        
        # In production, approval would be handled by UI/API layer
        # For now, auto-approve based on risk level
        if self.auto_approve:
            approved = True
            modified = None
        else:
            # Non-interactive mode: auto-approve medium/low, reject high
            if risk_level in ['low', 'medium']:
                approved = True
                modified = None
                logger.info(f"Auto-approved: {action} ({risk_level} risk)")
            else:
                approved = False
                modified = None
                logger.warning(f"Auto-rejected: {action} ({risk_level} risk - requires manual approval)")
        
        # Log decision
        self.approval_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'proposal': proposal,
            'approved': approved,
            'modified': modified is not None,
            'risk_level': risk_level
        })
        
        return approved, modified or proposal
    
    def _interactive_edit(self, proposal: Dict) -> Dict:
        """
        Edit proposal (stub for API-based editing).
        In production, this would be handled by the UI/API layer.
        """
        logger.info(f"Proposal edit requested: {proposal}")
        # Return unmodified - editing should be handled at API layer
        return proposal
    
    def clarify_ambiguous_query(self, query: str, options: List[str]) -> str:
        """
        Request clarification when query is ambiguous.
        
        Parameters
        ----------
        query : str
            Ambiguous query
        options : List[str]
            Possible interpretations
            
        Returns
        -------
        str
            Selected interpretation
        """
        logger.info(f"Clarification needed for query: '{query}'")
        logger.info(f"Options: {options}")
        
        # In production, clarification would be handled by UI/API layer
        # For now, default to first option
        selected = options[0]
        logger.info(f"Auto-selected: {selected}")
        return selected
    
    def iterative_refinement(
        self,
        initial_result: Dict,
        max_iterations: int = 3
    ) -> Dict:
        """
        Allow iterative refinement of results.
        
        In production, refinement would be handled by UI/API layer with separate requests.
        This is a stub for API-based iterative workflows.
        
        Parameters
        ----------
        initial_result : Dict
            Initial analysis result
        max_iterations : int, default=3
            Maximum refinement iterations
            
        Returns
        -------
        Dict
            Final refined result
        """
        logger.info(f"Refinement workflow initiated (max iterations: {max_iterations})")
        logger.debug(f"Initial result: {initial_result}")
        
        # In production, each refinement would be a separate API call
        # For now, return initial result
        return initial_result
    
    def get_approval_stats(self) -> Dict:
        """Get statistics on approval history"""
        if not self.approval_history:
            return {'total': 0, 'approved': 0, 'rejected': 0}
        
        total = len(self.approval_history)
        approved = sum(1 for h in self.approval_history if h['approved'])
        modified = sum(1 for h in self.approval_history if h['modified'])
        
        return {
            'total': total,
            'approved': approved,
            'rejected': total - approved,
            'modified': modified,
            'approval_rate': approved / total if total > 0 else 0.0
        }

