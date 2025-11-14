"""
Conversation Context Management

Tracks multi-turn dialogue history for contextual understanding.
"""

from typing import List, Dict, Optional
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationContext:
    """
    Manages conversation history and context for multi-turn dialogues.
    
    Features:
    - Session-based conversation tracking
    - Context window management (token limits)
    - Entity tracking (channels, metrics mentioned)
    - Conversation persistence
    
    Parameters
    ----------
    session_id : str, optional
        Unique session identifier. Generates UUID if not provided.
    max_history : int, default=20
        Maximum number of turns to keep in memory
    context_window : int, default=10
        Number of recent turns to include in context
    
    Examples
    --------
    >>> context = ConversationContext(session_id="user123")
    >>> context.add_turn("user", "What's TV's ROI?")
    >>> context.add_turn("assistant", "TV has an ROI of 1.32x")
    >>> context.add_turn("user", "How does it compare to Radio?")
    >>> 
    >>> # Resolve pronoun using context
    >>> context.get_last_mentioned_channel()
    'TV'
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        max_history: int = 20,
        context_window: int = 10,
        persistence_path: Optional[Path] = None
    ):
        import uuid
        
        self.session_id = session_id or str(uuid.uuid4())
        self.max_history = max_history
        self.context_window = context_window
        self.persistence_path = persistence_path or Path("logs/conversations")
        
        self.history: List[Dict] = []
        self.entities: Dict[str, List] = {
            'channels': [],
            'metrics': [],
            'timeframes': []
        }
        self.created_at = datetime.now()
        
        logger.info(f"Initialized ConversationContext: session={self.session_id}")
    
    def add_turn(
        self,
        role: str,
        content: str,
        *,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a conversation turn.
        
        Parameters
        ----------
        role : str
            'user' or 'assistant'
        content : str
            Message content
        metadata : Dict, optional
            Additional metadata (query_type, channels_mentioned, etc.)
        """
        turn = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.history.append(turn)
        
        # Extract and track entities
        if role == 'user':
            self._extract_entities(content)
        
        # Trim history if exceeds max
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        logger.debug(f"Added {role} turn: {content[:50]}...")
    
    def get_context(self, include_last_n: Optional[int] = None) -> List[Dict]:
        """
        Get conversation context for LLM.
        
        Parameters
        ----------
        include_last_n : int, optional
            Number of recent turns. Defaults to context_window.
            
        Returns
        -------
        List[Dict]
            List of conversation turns in OpenAI format
        """
        n = include_last_n or self.context_window
        recent_history = self.history[-n:] if len(self.history) > n else self.history
        
        return [
            {'role': turn['role'], 'content': turn['content']}
            for turn in recent_history
        ]
    
    def get_last_mentioned_channel(self) -> Optional[str]:
        """Get the most recently mentioned channel"""
        if self.entities['channels']:
            return self.entities['channels'][-1]
        return None
    
    def get_mentioned_entities(self, entity_type: str) -> List[str]:
        """
        Get all mentioned entities of a type.
        
        Parameters
        ----------
        entity_type : str
            'channels', 'metrics', or 'timeframes'
            
        Returns
        -------
        List[str]
            List of unique entities (most recent first)
        """
        entities = self.entities.get(entity_type, [])
        # Return unique entities in reverse order (most recent first)
        seen = set()
        unique = []
        for entity in reversed(entities):
            if entity not in seen:
                unique.append(entity)
                seen.add(entity)
        return unique
    
    def resolve_pronouns(self, query: str) -> str:
        """
        Resolve pronouns in query using context.
        
        Examples:
        - "it" -> "TV" (if TV was last mentioned)
        - "that channel" -> "Radio"
        
        Parameters
        ----------
        query : str
            User query with potential pronouns
            
        Returns
        -------
        str
            Query with pronouns resolved
        """
        resolved = query
        
        # Common pronoun patterns
        pronouns = {
            'it': self.get_last_mentioned_channel(),
            'that': self.get_last_mentioned_channel(),
            'that channel': self.get_last_mentioned_channel(),
            'this channel': self.get_last_mentioned_channel(),
        }
        
        for pronoun, entity in pronouns.items():
            if entity and pronoun in resolved.lower():
                resolved = resolved.replace(pronoun, entity)
                resolved = resolved.replace(pronoun.title(), entity)
                logger.info(f"Resolved pronoun '{pronoun}' to '{entity}'")
        
        return resolved
    
    def _extract_entities(self, text: str) -> None:
        """
        Extract entities from user message.
        
        Simple keyword-based extraction. Could be enhanced with NER.
        """
        text_lower = text.lower()
        
        # Common marketing channels
        channels = ['tv', 'radio', 'social', 'search', 'display', 'email', 
                   'video', 'print', 'ooh', 'outdoor', 'digital', 'facebook',
                   'google', 'instagram', 'tiktok', 'youtube', 'linkedin']
        
        for channel in channels:
            if channel in text_lower:
                self.entities['channels'].append(channel.upper())
        
        # Common MMM metrics
        metrics = ['roi', 'roas', 'cpa', 'cpm', 'saturation', 'adstock',
                  'contribution', 'spend', 'response', 'elasticity', 
                  'incrementality', 'baseline']
        
        for metric in metrics:
            if metric in text_lower:
                self.entities['metrics'].append(metric.upper())
        
        # Time periods
        timeframes = ['week', 'month', 'quarter', 'year', 'last week',
                     'last month', 'ytd', 'q1', 'q2', 'q3', 'q4']
        
        for timeframe in timeframes:
            if timeframe in text_lower:
                self.entities['timeframes'].append(timeframe)
    
    def save(self) -> None:
        """Save conversation to disk"""
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        filepath = self.persistence_path / f"{self.session_id}.json"
        
        data = {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'history': self.history,
            'entities': self.entities
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved conversation to {filepath}")
    
    @classmethod
    def load(cls, session_id: str, persistence_path: Optional[Path] = None) -> 'ConversationContext':
        """
        Load conversation from disk.
        
        Parameters
        ----------
        session_id : str
            Session ID to load
        persistence_path : Path, optional
            Directory containing conversation files
            
        Returns
        -------
        ConversationContext
            Loaded conversation context
        """
        persistence_path = persistence_path or Path("logs/conversations")
        filepath = persistence_path / f"{session_id}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"No conversation found for session {session_id}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        context = cls(session_id=data['session_id'], persistence_path=persistence_path)
        context.history = data['history']
        context.entities = data['entities']
        context.created_at = datetime.fromisoformat(data['created_at'])
        
        logger.info(f"Loaded conversation from {filepath}")
        return context
    
    def clear(self) -> None:
        """Clear conversation history"""
        self.history = []
        self.entities = {'channels': [], 'metrics': [], 'timeframes': []}
        logger.info(f"Cleared conversation context: session={self.session_id}")
    
    def get_summary(self) -> Dict:
        """Get conversation summary statistics"""
        return {
            'session_id': self.session_id,
            'num_turns': len(self.history),
            'duration_minutes': (datetime.now() - self.created_at).seconds / 60,
            'channels_discussed': len(set(self.entities['channels'])),
            'metrics_discussed': len(set(self.entities['metrics']))
        }

