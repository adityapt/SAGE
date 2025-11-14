"""
Knowledge Base for RAG Retrieval

Stores and retrieves MMM model outputs using semantic search.
Supports multiple backends: in-memory, ChromaDB, or file-based.
"""

from typing import Dict, List, Optional, Tuple, Literal
import logging
from pathlib import Path
import os

import numpy as np
import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Knowledge base for MMM results with semantic search.
    
    Stores model outputs (coefficients, ROI, saturation, etc.) and enables
    semantic search using OpenAI embeddings for RAG-based querying.
    
    Parameters
    ----------
    api_key : str
        OpenAI API key
    embedding_model : str, default="text-embedding-3-small"
        OpenAI embedding model
        
    Examples
    --------
    >>> kb = KnowledgeBase(api_key="...")
    >>> kb.add_document('coef_TV', 'TV coefficient: 1.32, ROI: 1.32x...')
    >>> docs = kb.search("What is TV's ROI?", top_k=3)
    """
    
    def __init__(
        self,
        api_key: str,
        *,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        
        self.documents: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        
        logger.info(f"Initialized KnowledgeBase with model={embedding_model}")
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add document to knowledge base.
        
        Parameters
        ----------
        doc_id : str
            Unique document identifier
        content : str
            Document text content
        metadata : Dict, optional
            Additional metadata (channel, type, etc.)
        """
        # Generate embedding
        embedding = self._get_embedding(content)
        
        # Store document
        self.documents.append({
            'id': doc_id,
            'content': content,
            'metadata': metadata or {}
        })
        self.embeddings.append(embedding)
        
        logger.debug(f"Added document: {doc_id}")
    
    def add_from_curves(self, curves: Dict[str, Dict]) -> None:
        """
        Populate knowledge base from response curves.
        
        Parameters
        ----------
        curves : Dict[str, Dict]
            Response curves from ResponseCurveGenerator
        """
        for channel, params in curves.items():
            content = f"""Channel: {channel}
Saturation Level: ${params['saturation']:,.0f}
Slope: {params['slope']:.3f}
Top Response: {params['top']:,.0f}
R-squared: {params['r_2']:.3f}

Interpretation: This channel has a saturation point at ${params['saturation']:,.0f} spend.
The slope of {params['slope']:.3f} indicates {'steep' if params['slope'] > 2 else 'moderate'} response.
"""
            self.add_document(
                f"curve_{channel}",
                content,
                metadata={'channel': channel, 'type': 'response_curve'}
            )
        
        logger.info(f"Added {len(curves)} response curves to knowledge base")
    
    def add_from_optimization(
        self,
        result_df: pd.DataFrame,
        total_response: float
    ) -> None:
        """
        Add optimization results to knowledge base.
        
        Parameters
        ----------
        result_df : pd.DataFrame
            Optimization results by channel
        total_response : float
            Total predicted response
        """
        # Overall optimization summary
        content = f"""Optimization Results:
Total Budget: ${result_df['total_spend'].sum():,.0f}
Total Response: {total_response:,.0f}
Overall ROI: {total_response / result_df['total_spend'].sum():.2f}x

Channel Allocations:
"""
        for _, row in result_df.iterrows():
            content += f"- {row['channel']}: ${row['total_spend']:,.0f} ({row['total_spend']/result_df['total_spend'].sum()*100:.1f}%)\n"
        
        self.add_document(
            'optimization_overall',
            content,
            metadata={'type': 'optimization'}
        )
        
        # Individual channel results
        for _, row in result_df.iterrows():
            content = f"""Optimization - {row['channel']}:
Allocated Spend: ${row['total_spend']:,.0f}
Expected Response: {row['total_response']:,.0f}
ROI: {row['roi']:.2f}x
Weekly Spend: ${row['weekly_spend']:,.0f}
"""
            self.add_document(
                f"opt_{row['channel']}",
                content,
                metadata={'channel': row['channel'], 'type': 'optimization'}
            )
        
        logger.info("Added optimization results to knowledge base")
    
    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search knowledge base using semantic similarity.
        
        Parameters
        ----------
        query : str
            Search query
        top_k : int, default=3
            Number of documents to return
        filter_metadata : Dict, optional
            Filter by metadata (e.g., {'type': 'response_curve'})
            
        Returns
        -------
        List[Dict]
            Top-k most relevant documents
        """
        if not self.documents:
            logger.warning("Knowledge base is empty")
            return []
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Apply metadata filter if specified
            if filter_metadata:
                match = all(
                    self.documents[i]['metadata'].get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue
            
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity and return top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_docs = [
            {**self.documents[idx], 'similarity': sim}
            for sim, idx in similarities[:top_k]
        ]
        
        logger.info(f"Retrieved {len(top_docs)} documents for query: {query[:50]}...")
        return top_docs
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI API"""
        try:
            # Ensure text is valid
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding, returning zero vector")
                return np.zeros(1536)
            
            # OpenAI API requires input as a list
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text.strip()]  # Must be a list
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector on failure
            return np.zeros(1536)  # text-embedding-3-small dimension
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)
    
    def clear(self) -> None:
        """Clear all documents from knowledge base"""
        self.documents = []
        self.embeddings = []
        logger.info("Cleared knowledge base")
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            'num_documents': len(self.documents),
            'document_types': pd.Series([
                doc['metadata'].get('type', 'unknown')
                for doc in self.documents
            ]).value_counts().to_dict()
        }


class ChromaKnowledgeBase(KnowledgeBase):
    """
    ChromaDB-backed knowledge base for persistent storage and efficient vector search.
    
    Advantages over in-memory:
    - Persistent storage (survives restarts)
    - No need to re-generate embeddings
    - Efficient for large document collections (>1000 docs)
    - Built-in filtering and metadata queries
    
    Parameters
    ----------
    api_key : str
        OpenAI API key (for embedding model)
    persist_directory : str, default="./chroma_db"
        Directory to store ChromaDB data
    collection_name : str, default="mmm_knowledge"
        Name of the ChromaDB collection
    embedding_model : str, default="text-embedding-3-small"
        OpenAI embedding model
        
    Examples
    --------
    >>> kb = ChromaKnowledgeBase(api_key="...", persist_directory="./knowledge_db")
    >>> kb.add_document('coef_TV', 'TV coefficient: 1.32, ROI: 1.32x...')
    >>> docs = kb.search("What is TV's ROI?", top_k=3)
    >>> 
    >>> # On restart, embeddings are already there!
    >>> kb = ChromaKnowledgeBase(api_key="...", persist_directory="./knowledge_db")
    >>> docs = kb.search("What is TV's ROI?", top_k=3)  # No re-embedding needed
    """
    
    def __init__(
        self,
        api_key: str,
        *,
        persist_directory: str = "./chroma_db",
        collection_name: str = "mmm_knowledge",
        embedding_model: str = "text-embedding-3-small"
    ):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize OpenAI client (for embeddings)
        self.client = OpenAI(api_key=api_key)
        
        # Initialize ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection with custom embedding function
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "MMM knowledge base with semantic search"}
        )
        
        logger.info(
            f"Initialized ChromaKnowledgeBase: "
            f"persist_dir={persist_directory}, "
            f"collection={collection_name}, "
            f"existing_docs={self.collection.count()}"
        )
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add document to ChromaDB knowledge base.
        
        Parameters
        ----------
        doc_id : str
            Unique document identifier
        content : str
            Document text content
        metadata : Dict, optional
            Additional metadata (channel, type, etc.)
        """
        # Generate embedding using OpenAI
        embedding = self._get_embedding(content)
        
        # Upsert to ChromaDB (add or update if exists)
        self.collection.upsert(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding.tolist()],
            metadatas=[metadata or {}]
        )
        
        logger.debug(f"Upserted document to ChromaDB: {doc_id}")
    
    def add_from_curves(self, curves: Dict[str, Dict]) -> None:
        """Populate knowledge base from response curves"""
        for channel, params in curves.items():
            content = f"""Channel: {channel}
Saturation Level: ${params['saturation']:,.0f}
Slope: {params['slope']:.3f}
Top Response: {params['top']:,.0f}
R-squared: {params['r_2']:.3f}

Interpretation: This channel has a saturation point at ${params['saturation']:,.0f} spend.
The slope of {params['slope']:.3f} indicates {'steep' if params['slope'] > 2 else 'moderate'} response.
"""
            self.add_document(
                f"curve_{channel}",
                content,
                metadata={'channel': channel, 'type': 'response_curve'}
            )
        
        logger.info(f"Added {len(curves)} response curves to ChromaDB")
    
    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search knowledge base using semantic similarity.
        
        Parameters
        ----------
        query : str
            Search query
        top_k : int, default=3
            Number of documents to return
        filter_metadata : Dict, optional
            Filter by metadata (e.g., {'type': 'response_curve'})
            
        Returns
        -------
        List[Dict]
            Top-k most relevant documents
        """
        if self.collection.count() == 0:
            logger.warning("ChromaDB collection is empty")
            return []
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata if filter_metadata else None,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results to match in-memory format
        docs = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # Convert distance to similarity (ChromaDB returns L2 distance)
                # similarity = 1 / (1 + distance)
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance)
                
                docs.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': similarity
                })
        
        logger.info(f"Retrieved {len(docs)} documents from ChromaDB for query: {query[:50]}...")
        return docs
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI API"""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding, returning zero vector")
                return np.zeros(1536)
            
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text.strip()]
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros(1536)
    
    def clear(self) -> None:
        """Clear all documents from ChromaDB collection"""
        # Delete and recreate collection
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "MMM knowledge base with semantic search"}
        )
        logger.info("Cleared ChromaDB collection")
    
    def get_stats(self) -> Dict:
        """Get ChromaDB collection statistics"""
        count = self.collection.count()
        
        # Get all metadatas to count types
        if count > 0:
            all_docs = self.collection.get(include=['metadatas'])
            doc_types = pd.Series([
                meta.get('type', 'unknown')
                for meta in all_docs['metadatas']
            ]).value_counts().to_dict()
        else:
            doc_types = {}
        
        return {
            'backend': 'chromadb',
            'persist_directory': self.persist_directory,
            'collection_name': self.collection_name,
            'num_documents': count,
            'document_types': doc_types
        }


def create_knowledge_base(
    api_key: str,
    backend: Literal['memory', 'chromadb'] = 'memory',
    **kwargs
) -> KnowledgeBase:
    """
    Factory function to create knowledge base with specified backend.
    
    Parameters
    ----------
    api_key : str
        OpenAI API key
    backend : {'memory', 'chromadb'}, default='memory'
        Storage backend
    **kwargs
        Additional arguments passed to backend implementation
        
    Returns
    -------
    KnowledgeBase
        Knowledge base instance
        
    Examples
    --------
    >>> # In-memory (default, lost on restart)
    >>> kb = create_knowledge_base(api_key="...", backend='memory')
    >>> 
    >>> # ChromaDB (persistent, survives restarts)
    >>> kb = create_knowledge_base(
    ...     api_key="...",
    ...     backend='chromadb',
    ...     persist_directory='./knowledge_db'
    ... )
    """
    if backend == 'memory':
        return KnowledgeBase(api_key=api_key, **kwargs)
    elif backend == 'chromadb':
        return ChromaKnowledgeBase(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'memory' or 'chromadb'.")

