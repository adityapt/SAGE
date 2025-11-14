"""
Advanced RAG Features

Query rewriting, confidence scoring, citations, multi-hop reasoning.
"""

from typing import List, Dict, Optional, Tuple
import logging
import re

from openai import OpenAI

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Query rewriting for improved retrieval.
    
    Techniques:
    - Expand acronyms (ROI -> Return on Investment)
    - Add synonyms (spend -> budget, cost)
    - Decompose complex queries into sub-queries
    - Resolve ambiguities
    
    Parameters
    ----------
    client : OpenAI
        OpenAI client
    
    Examples
    --------
    >>> rewriter = QueryRewriter(client)
    >>> rewrites = rewriter.rewrite("What's TV ROI?")
    >>> # Returns: ["What is TV Return on Investment?", "TV channel ROI", ...]
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def rewrite(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate query variants for better retrieval.
        
        Parameters
        ----------
        query : str
            Original query
        num_variants : int, default=3
            Number of variants to generate
            
        Returns
        -------
        List[str]
            Query variants
        """
        prompt = f"""You are a query rewriting expert for Marketing Mix Modeling.

Original query: "{query}"

Generate {num_variants} alternative phrasings that would retrieve the same information.

Rules:
- Expand acronyms (ROI -> Return on Investment, ROAS -> Return on Ad Spend)
- Add synonyms (spend -> budget, cost; channel -> medium)
- Make implicit context explicit
- Keep the core intent

Return ONLY the rewritten queries, one per line, without numbering."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            variants = [line.strip() for line in content.split('\n') if line.strip()]
            
            logger.info(f"Generated {len(variants)} query variants")
            return variants[:num_variants]
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return [query]  # Fallback to original
    
    def decompose(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries.
        
        Example: "Compare TV and Radio ROI and suggest optimization"
        -> ["What is TV ROI?", "What is Radio ROI?", "How to optimize budget?"]
        """
        prompt = f"""You are a query decomposition expert for Marketing Mix Modeling.

Original query: "{query}"

If this is a complex query with multiple questions, break it into simple sub-queries.
If it's already simple, return just the original query.

Return ONLY the sub-queries, one per line."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            subqueries = [line.strip() for line in content.split('\n') if line.strip()]
            
            logger.info(f"Decomposed into {len(subqueries)} sub-queries")
            return subqueries
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]


class ConfidenceScorer:
    """
    Calculate confidence scores for answers.
    
    Factors:
    - Source quality (data > benchmark > general knowledge)
    - Retrieval similarity scores
    - Answer consistency
    - Uncertainty markers in response
    
    Examples
    --------
    >>> scorer = ConfidenceScorer()
    >>> score = scorer.calculate(
    ...     answer="TV has ROI of 1.32x",
    ...     sources=[{'similarity': 0.95, 'type': 'data'}],
    ...     query="What is TV ROI?"
    ... )
    >>> score  # 0.92
    """
    
    def __init__(self):
        # Uncertainty markers that lower confidence
        self.uncertainty_markers = [
            'might', 'may', 'possibly', 'likely', 'probably',
            'uncertain', 'not sure', 'unclear', 'approximately',
            'roughly', 'around', 'about', 'seems', 'appears'
        ]
    
    def calculate(
        self,
        answer: str,
        sources: List[Dict],
        query: str
    ) -> float:
        """
        Calculate confidence score (0-1).
        
        Parameters
        ----------
        answer : str
            Generated answer
        sources : List[Dict]
            Retrieved sources with similarity scores
        query : str
            Original query
            
        Returns
        -------
        float
            Confidence score between 0 and 1
        """
        if not sources:
            return 0.0
        
        # Factor 1: Average retrieval similarity (40%)
        avg_similarity = sum(s.get('similarity', 0) for s in sources) / len(sources)
        retrieval_score = avg_similarity * 0.4
        
        # Factor 2: Source quality (30%)
        source_quality = self._score_source_quality(sources)
        quality_score = source_quality * 0.3
        
        # Factor 3: Answer confidence (20%)
        answer_confidence = self._score_answer_confidence(answer)
        answer_score = answer_confidence * 0.2
        
        # Factor 4: Query-answer alignment (10%)
        alignment = self._score_alignment(query, answer)
        alignment_score = alignment * 0.1
        
        total_score = retrieval_score + quality_score + answer_score + alignment_score
        
        logger.debug(f"Confidence: {total_score:.2f} (retrieval={retrieval_score:.2f}, quality={quality_score:.2f}, answer={answer_score:.2f}, alignment={alignment_score:.2f})")
        
        return round(total_score, 2)
    
    def _score_source_quality(self, sources: List[Dict]) -> float:
        """Score based on source types"""
        quality_weights = {
            'data': 1.0,           # Actual model data
            'response_curve': 0.9, # Fitted curves
            'optimization': 0.9,   # Optimization results
            'benchmark': 0.7,      # Industry benchmarks
            'best_practice': 0.6,  # General best practices
            'unknown': 0.3
        }
        
        if not sources:
            return 0.0
        
        avg_quality = sum(
            quality_weights.get(s.get('metadata', {}).get('type', 'unknown'), 0.3)
            for s in sources
        ) / len(sources)
        
        return avg_quality
    
    def _score_answer_confidence(self, answer: str) -> float:
        """Score based on answer characteristics"""
        answer_lower = answer.lower()
        
        # Check for uncertainty markers
        uncertainty_count = sum(
            1 for marker in self.uncertainty_markers
            if marker in answer_lower
        )
        
        # Penalize uncertainty
        uncertainty_penalty = min(uncertainty_count * 0.15, 0.5)
        
        # Check for specific numbers (higher confidence)
        has_numbers = bool(re.search(r'\d+\.?\d*', answer))
        number_bonus = 0.2 if has_numbers else 0.0
        
        # Check for hedging phrases
        hedging_phrases = ['it depends', 'varies', 'could be', 'it\'s hard to say']
        has_hedging = any(phrase in answer_lower for phrase in hedging_phrases)
        hedging_penalty = 0.2 if has_hedging else 0.0
        
        score = 1.0 - uncertainty_penalty + number_bonus - hedging_penalty
        return max(0.0, min(1.0, score))
    
    def _score_alignment(self, query: str, answer: str) -> float:
        """Score query-answer alignment using keyword overlap"""
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Extract key terms (simple approach)
        query_terms = set(re.findall(r'\b\w{4,}\b', query_lower))
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer_lower))
        
        if not query_terms:
            return 0.5
        
        overlap = len(query_terms & answer_terms)
        alignment = overlap / len(query_terms)
        
        return min(1.0, alignment)


class CitationManager:
    """
    Manage citations and source attribution.
    
    Tracks:
    - Source documents
    - Retrieved passages
    - Citation links
    - Confidence scores
    
    Examples
    --------
    >>> citations = CitationManager()
    >>> citations.add_source("data", "TV ROI: 1.32x from model", 0.95)
    >>> citations.format_citations()
    '[1] TV ROI: 1.32x from model (confidence: 95%)'
    """
    
    def __init__(self):
        self.sources: List[Dict] = []
    
    def add_source(
        self,
        source_type: str,
        content: str,
        similarity: float,
        *,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a source citation.
        
        Returns
        -------
        int
            Citation ID
        """
        citation_id = len(self.sources) + 1
        
        self.sources.append({
            'id': citation_id,
            'type': source_type,
            'content': content,
            'similarity': similarity,
            'metadata': metadata or {}
        })
        
        return citation_id
    
    def format_citations(self, format_type: str = "numbered") -> str:
        """
        Format citations for display.
        
        Parameters
        ----------
        format_type : str, default="numbered"
            Citation format: "numbered", "apa", "harvard"
            
        Returns
        -------
        str
            Formatted citations
        """
        if not self.sources:
            return "No sources"
        
        if format_type == "numbered":
            lines = []
            for source in self.sources:
                lines.append(
                    f"[{source['id']}] {source['content'][:100]}... "
                    f"(type: {source['type']}, confidence: {source['similarity']*100:.0f}%)"
                )
            return "\n".join(lines)
        
        elif format_type == "inline":
            # For inline citations like: "TV ROI is 1.32x [1]"
            return ", ".join(f"[{s['id']}]" for s in self.sources)
        
        else:
            return f"{len(self.sources)} sources cited"
    
    def get_sources(self) -> List[Dict]:
        """Get all sources"""
        return self.sources
    
    def clear(self) -> None:
        """Clear all citations"""
        self.sources = []


class MultiHopReasoner:
    """
    Multi-hop reasoning for complex queries.
    
    Chains multiple retrieval + reasoning steps.
    
    Example:
    Query: "Which channel has better ROI, TV or Radio, and why?"
    Step 1: Retrieve TV ROI
    Step 2: Retrieve Radio ROI
    Step 3: Retrieve benchmark context
    Step 4: Synthesize comparison
    """
    
    def __init__(self, client: OpenAI, knowledge_base):
        self.client = client
        self.knowledge_base = knowledge_base
    
    def reason(self, query: str, max_hops: int = 3) -> Tuple[str, List[Dict]]:
        """
        Perform multi-hop reasoning.
        
        Parameters
        ----------
        query : str
            Complex query
        max_hops : int, default=3
            Maximum reasoning steps
            
        Returns
        -------
        Tuple[str, List[Dict]]
            Final answer and reasoning trace
        """
        trace = []
        context = ""
        
        for hop in range(max_hops):
            logger.info(f"Reasoning hop {hop + 1}")
            
            # Generate sub-query
            if hop == 0:
                subquery = query
            else:
                subquery = self._generate_followup_query(query, context)
            
            # Retrieve
            docs = self.knowledge_base.search(subquery, top_k=3)
            
            # Update context
            new_context = "\n".join(d['content'] for d in docs)
            context += f"\n\nHop {hop + 1}:\n{new_context}"
            
            trace.append({
                'hop': hop + 1,
                'subquery': subquery,
                'retrieved': len(docs),
                'context_length': len(context)
            })
            
            # Check if sufficient
            if self._is_sufficient(query, context):
                logger.info(f"Sufficient context after {hop + 1} hops")
                break
        
        # Generate final answer
        answer = self._synthesize_answer(query, context)
        
        return answer, trace
    
    def _generate_followup_query(self, original_query: str, context: str) -> str:
        """Generate follow-up query based on context"""
        prompt = f"""Given the original query and context so far, what additional information is needed?

Original query: {original_query}

Context so far:
{context[:500]}...

Generate a focused follow-up question to retrieve missing information.
Return ONLY the question."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Follow-up query generation failed: {e}")
            return original_query
    
    def _is_sufficient(self, query: str, context: str) -> bool:
        """Check if context is sufficient to answer query"""
        # Simple heuristic: check if context has key terms from query
        query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
        context_terms = set(re.findall(r'\b\w{4,}\b', context.lower()))
        
        coverage = len(query_terms & context_terms) / len(query_terms) if query_terms else 0
        return coverage > 0.7
    
    def _synthesize_answer(self, query: str, context: str) -> str:
        """Synthesize final answer from multi-hop context"""
        prompt = f"""Based on the retrieved context, answer the query comprehensively.

Query: {query}

Context:
{context}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return "Unable to synthesize answer from context."

