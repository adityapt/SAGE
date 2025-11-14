"""
Configuration Management

Loads configuration from environment variables and provides type-safe access.
"""

import os
from typing import Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
except ImportError:
    logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")


class Config:
    """
    Configuration management with environment variables.
    
    Supports:
    - Environment variable loading
    - Type conversion
    - Default values
    - Validation
    
    Examples
    --------
    >>> config = Config()
    >>> config.openai_api_key
    'sk-...'
    >>> config.cache_enabled
    True
    """
    
    # OpenAI Configuration
    @property
    def openai_api_key(self) -> str:
        key = os.getenv('OPENAI_API_KEY', '')
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not set. "
                "Set it in .env file or export OPENAI_API_KEY=your_key"
            )
        return key
    
    @property
    def openai_base_url(self) -> str:
        """OpenAI API base URL - supports custom proxies"""
        return os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    
    @property
    def openai_embedding_model(self) -> str:
        return os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    
    @property
    def openai_chat_model(self) -> str:
        return os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o')
    
    # Web Search Configuration
    @property
    def web_search_provider(self) -> str:
        """Web search provider: tavily, brave, or serper"""
        return os.getenv('WEB_SEARCH_PROVIDER', 'tavily')
    
    @property
    def tavily_api_key(self) -> Optional[str]:
        """Tavily API key (best for LLM/RAG use cases)"""
        return os.getenv('TAVILY_API_KEY')
    
    @property
    def brave_api_key(self) -> Optional[str]:
        """Brave Search API key (privacy-focused)"""
        return os.getenv('BRAVE_API_KEY')
    
    @property
    def serper_api_key(self) -> Optional[str]:
        """Serper API key (Google Search proxy)"""
        return os.getenv('SERPER_API_KEY')
    
    @property
    def enable_web_search(self) -> bool:
        """Enable web search fallback for out-of-scope queries"""
        return os.getenv('ENABLE_WEB_SEARCH', 'true').lower() == 'true'
    
    @property
    def confidence_threshold(self) -> float:
        """Confidence threshold for triggering web search (0.0-1.0)"""
        return float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))
    
    # Database Configuration
    @property
    def database_url(self) -> Optional[str]:
        return os.getenv('DATABASE_URL')
    
    @property
    def database_schema(self) -> str:
        return os.getenv('DATABASE_SCHEMA', 'public')
    
    # Cache Configuration
    @property
    def redis_url(self) -> Optional[str]:
        return os.getenv('REDIS_URL')
    
    @property
    def cache_enabled(self) -> bool:
        return os.getenv('REDIS_URL') is not None
    
    @property
    def cache_ttl_seconds(self) -> int:
        return int(os.getenv('CACHE_TTL_SECONDS', '3600'))
    
    # Logging Configuration
    @property
    def log_level(self) -> str:
        return os.getenv('LOG_LEVEL', 'INFO')
    
    @property
    def metrics_enabled(self) -> bool:
        return os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
    
    @property
    def trace_api_calls(self) -> bool:
        return os.getenv('TRACE_API_CALLS', 'false').lower() == 'true'
    
    # API Server Configuration
    @property
    def api_host(self) -> str:
        return os.getenv('API_HOST', '0.0.0.0')
    
    @property
    def api_port(self) -> int:
        return int(os.getenv('API_PORT', '8000'))
    
    @property
    def api_workers(self) -> int:
        return int(os.getenv('API_WORKERS', '4'))
    
    # Security
    @property
    def api_key_required(self) -> bool:
        return os.getenv('API_KEY_REQUIRED', 'false').lower() == 'true'
    
    @property
    def api_keys(self) -> List[str]:
        keys_str = os.getenv('API_KEYS', '')
        return [k.strip() for k in keys_str.split(',') if k.strip()]
    
    # Performance
    @property
    def max_workers(self) -> int:
        return int(os.getenv('MAX_WORKERS', '4'))
    
    @property
    def embedding_batch_size(self) -> int:
        return int(os.getenv('EMBEDDING_BATCH_SIZE', '100'))
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns
        -------
        bool
            True if valid, raises ValueError otherwise
        """
        # Check critical settings
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        if self.cache_ttl_seconds < 0:
            raise ValueError("CACHE_TTL_SECONDS must be >= 0")
        
        if self.api_port < 1024 or self.api_port > 65535:
            raise ValueError("API_PORT must be between 1024 and 65535")
        
        logger.info("Configuration validated successfully")
        return True
    
    def __repr__(self) -> str:
        """Safe representation without exposing secrets"""
        return (
            f"Config("
            f"embedding_model={self.openai_embedding_model}, "
            f"chat_model={self.openai_chat_model}, "
            f"cache_enabled={self.cache_enabled}, "
            f"metrics_enabled={self.metrics_enabled}"
            f")"
        )


# Global configuration instance
config = Config()

