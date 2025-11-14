"""
Agentic Tool Calling Framework

Enables LLM to decide which external tools to use based on query needs.
"""

from typing import Dict, List, Optional, Callable, Any
import logging
import json

logger = logging.getLogger(__name__)


class Tool:
    """
    Represents a callable tool that the LLM can invoke.
    
    Parameters
    ----------
    name : str
        Tool name (must be unique)
    description : str
        What the tool does (LLM uses this to decide when to call)
    function : Callable
        The actual function to execute
    parameters : Dict
        JSON schema describing function parameters
    
    Examples
    --------
    >>> def search_web(query: str) -> str:
    ...     return f"Results for {query}"
    >>> tool = Tool(
    ...     name="search_web",
    ...     description="Search internet for marketing trends",
    ...     function=search_web,
    ...     parameters={
    ...         "type": "object",
    ...         "properties": {"query": {"type": "string"}},
    ...         "required": ["query"]
    ...     }
    ... )
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Dict
    ):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        try:
            result = self.function(**kwargs)
            logger.info(f"Tool '{self.name}' executed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}")
            raise
    
    def to_openai_format(self) -> Dict:
        """Convert tool to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Handles tool registration, execution, and LLM tool calling workflow.
    
    Examples
    --------
    >>> registry = ToolRegistry()
    >>> registry.register(search_tool)
    >>> registry.register(benchmark_tool)
    >>> 
    >>> # LLM decides to call search_web
    >>> result = registry.execute("search_web", query="CTV advertising trends")
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool"""
        if tool.name in self.tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name with parameters"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        return tool.execute(**kwargs)
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        Get all tools in OpenAI function calling format.
        
        Returns
        -------
        List[Dict]
            Tool definitions for OpenAI API
        """
        return [tool.to_openai_format() for tool in self.tools.values()]
    
    def list_tools(self) -> List[str]:
        """List names of all registered tools"""
        return list(self.tools.keys())


class AgenticToolCaller:
    """
    Agentic tool calling orchestrator.
    
    Determines if external tools are needed and executes them.
    
    Parameters
    ----------
    api_key : str
        OpenAI API key
    base_url : str, optional
        OpenAI API base URL
    tool_registry : ToolRegistry
        Registry of available tools
    
    Examples
    --------
    >>> caller = AgenticToolCaller(api_key="...", tool_registry=registry)
    >>> result = caller.process_query(
    ...     query="What are emerging marketing channels?",
    ...     context="Current channels: TV, Search, Social"
    ... )
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.registry = tool_registry or ToolRegistry()
    
    def process_query(
        self,
        query: str,
        context: str,
        *,
        max_iterations: int = 3
    ) -> Dict:
        """
        Process query with agentic tool calling.
        
        LLM decides if tools are needed and which ones to call.
        
        Parameters
        ----------
        query : str
            User query
        context : str
            Existing context (MMM data, etc.)
        max_iterations : int, default=3
            Maximum tool calling iterations
            
        Returns
        -------
        Dict
            Result with answer, tool_calls, and citations
        """
        messages = [
            {
                "role": "system",
                "content": """You are an MMM (Marketing Mix Modeling) expert assistant.

You have access to external tools for information you don't have in the provided context.

Decision criteria:
- If query is about data IN the context → answer directly
- If query needs external/real-time info → use tools
- If query is about emerging trends, benchmarks, or new channels → use search_web
- If query needs industry benchmarks → use get_industry_benchmark

Always cite sources when using external data."""
            },
            {
                "role": "user",
                "content": f"""Context: {context}

Query: {query}"""
            }
        ]
        
        tool_calls_made = []
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Call LLM with tools
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.registry.get_tool_definitions() if self.registry.tools else None,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # Check if LLM wants to use tools
            if not message.tool_calls:
                # No tools needed, return final answer
                return {
                    'answer': message.content,
                    'tool_calls': tool_calls_made,
                    'requires_tools': len(tool_calls_made) > 0
                }
            
            # Execute tool calls
            messages.append(message)
            
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"LLM requesting tool: {tool_name} with args: {tool_args}")
                
                try:
                    # Execute tool
                    tool_result = self.registry.execute(tool_name, **tool_args)
                    
                    # Track tool call
                    tool_calls_made.append({
                        'tool': tool_name,
                        'args': tool_args,
                        'result': tool_result[:500] if isinstance(tool_result, str) else str(tool_result)[:500]
                    })
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
                    
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: {str(e)}"
                    })
        
        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached in tool calling")
        return {
            'answer': "I encountered an issue processing your query with multiple tool calls.",
            'tool_calls': tool_calls_made,
            'requires_tools': True,
            'error': 'max_iterations_exceeded'
        }

