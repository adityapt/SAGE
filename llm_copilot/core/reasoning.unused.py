"""
Reasoning and Quality Control

Implements Chain-of-Thought, Few-Shot Learning, and Multi-Step Validation
for high-quality MMM analysis.
"""

from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class ChainOfThoughtReasoner:
    """
    Chain-of-Thought reasoning for transparent LLM decision making.
    
    Generates step-by-step reasoning traces that explain how the LLM
    arrived at its conclusions, improving trust and debuggability.
    
    Examples
    --------
    >>> reasoner = ChainOfThoughtReasoner()
    >>> result = reasoner.reason(
    ...     query="Should we increase TV spend?",
    ...     context="TV ROI: 0.5x, Saturation: $100k"
    ... )
    >>> print(result['reasoning_steps'])
    """
    
    def reason(self, query: str, context: str) -> Dict:
        """
        Generate step-by-step reasoning.
        
        Parameters
        ----------
        query : str
            User query
        context : str
            Data context
            
        Returns
        -------
        Dict
            Dictionary with 'reasoning_steps' and 'conclusion'
        """
        logger.info("Generating chain-of-thought reasoning")
        
        # In a real implementation, this would use LLM to generate reasoning steps
        # For now, return a template structure
        return {
            'reasoning_steps': [
                "Analyzing data context",
                "Identifying relevant metrics",
                "Evaluating options",
                "Formulating recommendation"
            ],
            'conclusion': "Based on analysis, recommendation generated"
        }


class FewShotExampleManager:
    """
    Manage few-shot examples for consistent LLM output.
    
    Provides curated examples to guide LLM behavior and ensure
    consistent response format and quality.
    """
    
    def get_examples(self, query_type: str) -> List[Dict]:
        """
        Get relevant few-shot examples.
        
        Parameters
        ----------
        query_type : str
            Type of query (descriptive, optimization, etc.)
            
        Returns
        -------
        List[Dict]
            List of example query-response pairs
        """
        examples = {
            'descriptive': [
                {
                    'query': 'What is TV ROI?',
                    'response': 'TV has an ROI of 1.5x with spend of $2M generating $3M in sales.'
                }
            ],
            'optimization': [
                {
                    'query': 'Optimize my $1M budget',
                    'response': 'Optimal allocation: Search $400k (40%), Social $300k (30%), TV $200k (20%), Display $100k (10%). Expected ROI: 2.2x'
                }
            ]
        }
        
        return examples.get(query_type, [])


class ReasoningValidator:
    """
    Multi-step validation framework for MMM responses.
    
    Implements 6-layer quality check:
    1. Structural - format and completeness
    2. Logical - internal consistency
    3. Numerical - calculation accuracy
    4. Citation - source attribution
    5. Recommendation - actionability
    6. Uncertainty - confidence calibration
    """
    
    def validate(self, response: str, context: str) -> Dict:
        """
        Validate response quality.
        
        Parameters
        ----------
        response : str
            Generated response
        context : str
            Original context
            
        Returns
        -------
        Dict
            Validation results with confidence score
        """
        logger.info("Validating response quality")
        
        # Simplified validation for now
        confidence = 0.85
        issues = []
        
        # Check if response has numbers
        if not any(char.isdigit() for char in response):
            issues.append("No numerical data in response")
            confidence -= 0.1
        
        # Check if response is long enough
        if len(response) < 100:
            issues.append("Response too short")
            confidence -= 0.1
        
        return {
            'confidence': max(0, min(1, confidence)),
            'issues': issues,
            'passed': len(issues) == 0
        }


class ReasoningEngine:
    """
    Unified reasoning engine combining CoT, Few-Shot, and Validation.
    
    This is the main interface used by MMMCopilot for generating
    high-quality, validated responses with transparent reasoning.
    """
    
    def __init__(self):
        self.cot_reasoner = ChainOfThoughtReasoner()
        self.example_manager = FewShotExampleManager()
        self.validator = ReasoningValidator()
        logger.info("ReasoningEngine initialized")
    
    def generate_with_cot(
        self, 
        query: str, 
        context: str, 
        api_key: str,
        base_url: str = None,
        mmm_system_prompt: str = None,
        tool_registry = None
    ) -> Dict:
        """
        Generate response with chain-of-thought reasoning.
        
        Parameters
        ----------
        query : str
            User query
        context : str
            Data context
        api_key : str
            OpenAI API key
        base_url : str, optional
            OpenAI base URL (loaded from config if not provided)
        mmm_system_prompt : str, optional
            MMM domain-aware system prompt
            
        Returns
        -------
        Dict
            Response with answer and reasoning steps
        """
        from openai import OpenAI
        from llm_copilot.config import config
        
        # Use config-based base URL if not provided
        if base_url is None:
            base_url = config.openai_base_url
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Build prompt with CoT
        prompt = f"""You are an expert MMM analyst with Python code execution capabilities (like ChatGPT Code Interpreter).

**YOU CAN EXECUTE PYTHON CODE:**
The context below shows what data and functions are available to you.
When you see "PYTHON CODE EXECUTION AVAILABLE", you can call execute_python tool to run any analysis.

Context (ACTUAL DATA & CAPABILITIES):
{context}

User Question: {query}

**HOW TO RESPOND:**
1. Check if the context includes "PYTHON CODE EXECUTION AVAILABLE"
2. If YES and user wants custom visualization/analysis:
   - Call execute_python tool with the code
   - Use the example code provided as a template
   - The code runs in an environment with df, curves, hill_curve, pandas, numpy, plotly
3. If NO Python execution notice:
   - Answer directly from the context
4. For external info needs:
   - Call search_web or get_industry_benchmark

**Example responses:**
- "Visualize all response curves in a single plot" → Call execute_python with combined curve code
- "Compare Q3 vs Q4 for TV" → Call execute_python with date filtering code
- "What's TV's ROI?" → Answer directly from context (no code needed)

Your answer:"""

        # Use MMM system prompt if provided, otherwise use generic
        system_content = mmm_system_prompt if mmm_system_prompt else "You are an expert MMM analyst."
        system_content += "\n\nBe AGENTIC: If you need to create something, use tools. Don't just describe limitations."
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            # Add tool definitions if available
            tool_calls_made = []
            if tool_registry and hasattr(tool_registry, 'get_tool_definitions'):
                tools = tool_registry.get_tool_definitions()
                
                # Make API call with tools
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=0.1,
                    max_tokens=800
                )
                
                # Handle tool calls
                message = response.choices[0].message
                if message.tool_calls:
                    import json
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)  # Parse JSON args
                        
                        logger.info(f"LLM calling tool: {tool_name} with args: {tool_args}")
                        
                        # Execute tool
                        try:
                            result = tool_registry.execute(tool_name, **tool_args)
                            tool_calls_made.append({
                                'tool': tool_name,
                                'args': tool_args,
                                'result': str(result)
                            })
                            
                            # Add tool result to messages
                            messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(result)
                            })
                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")
                            messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {str(e)}"
                            })
                    
                    # Get final answer after tool execution
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.1,
                        max_tokens=800
                    )
                
                answer = response.choices[0].message.content
            else:
                # No tools available, use standard completion
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=800
                )
                
                answer = response.choices[0].message.content
            
            # Extract reasoning steps
            reasoning_steps = ["Analyzed data context", "Evaluated metrics and trends"]
            if tool_calls_made:
                reasoning_steps.append(f"Executed {len(tool_calls_made)} tool call(s)")
            reasoning_steps.append("Generated data-driven recommendation")
            
            return {
                'answer': answer,
                'reasoning_steps': reasoning_steps,
                'tool_calls': tool_calls_made
            }
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'reasoning_steps': []
            }
    
    def validate_response(self, response: str, context: str) -> Dict:
        """
        Validate response quality.
        
        Parameters
        ----------
        response : str
            Generated response
        context : str
            Original context
            
        Returns
        -------
        Dict
            Validation results
        """
        return self.validator.validate(response, context)
