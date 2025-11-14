"""
Agentic MMM Analysis System

Inspired by autonomous data analysis systems, this module enables:
- Autonomous code generation and execution
- Multi-step planning
- Tool-based analysis
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class AgenticAnalyzer:
    """
    Autonomous analyzer that can:
    1. Plan analysis steps
    2. Write and execute Python code
    3. Generate visualizations
    4. Synthesize insights
    """
    
    def __init__(self, data: pd.DataFrame, api_key: str, base_url: Optional[str] = None, date_col: str = 'date'):
        self.data = data
        self.api_key = api_key
        self.base_url = base_url
        self.date_col = date_col
        self.curves = {}  # Will be fitted on first query
        
        # Allow custom base_url for enterprise deployments
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        
        # Initialize Knowledge Base for RAG with ChromaDB backend
        from llm_copilot.core.knowledge_base import ChromaKnowledgeBase
        from llm_copilot.knowledge.mmm_expertise import MMMExpertise
        import os
        
        # Use ChromaDB for persistent storage
        persist_dir = os.getenv('CHROMADB_PATH', './chroma_mmm_knowledge')
        self.knowledge_base = ChromaKnowledgeBase(
            api_key=api_key,
            persist_directory=persist_dir,
            collection_name='mmm_knowledge'
        )
        self.expertise = MMMExpertise()
        logger.info(f"Initialized ChromaDB Knowledge Base (persist_dir={persist_dir}) and MMM Expertise for RAG")
        
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query autonomously by:
        1. Deciding if code is needed
        2. Writing code if needed
        3. Executing code
        4. Synthesizing answer
        """
        logger.info(f"Agentic analysis: {query}")
        
        # Ensure knowledge base is populated and curves are fitted from current data
        # IMPORTANT: Curves must be re-fitted whenever data changes to keep ChromaDB in sync
        if not self.curves:
            kb_stats = self.knowledge_base.get_stats()
            
            # Always fit curves from current data
            from llm_copilot.core.response_curves import ResponseCurveGenerator
            curve_gen = ResponseCurveGenerator(self.data, date_col=self.date_col)
            
            if 'channel' in self.data.columns:
                channels = self.data['channel'].unique().tolist()
                self.curves = curve_gen.fit_curves(channels=channels)
                logger.info(f"Fitted {len(self.curves)} curves from current data")
                
                # Update ChromaDB with fresh curves (upsert = add or update)
                if kb_stats.get('total_docs', 0) == 0:
                    logger.info("Knowledge base empty - populating with curves, benchmarks, glossary, best practices...")
                    self._fit_curves()  # Full population (curves + benchmarks + glossary + practices)
                else:
                    logger.info(f"Knowledge base has {kb_stats['total_docs']} docs - updating curves only...")
                    # Update just the curves (benchmarks/glossary/practices don't change)
                    self.knowledge_base.add_from_curves(self.curves)
                    logger.info(f"Updated {len(self.curves)} response curves in ChromaDB (upserted)")
        
        # Define tools
        channels_list = self.data['channel'].unique().tolist() if 'channel' in self.data.columns else []
        date_col = self.date_col
        sample_dates = self.data[date_col].head(3).tolist() if date_col in self.data.columns else []
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": f"""Execute Python code to analyze data. 

CHAIN-OF-THOUGHT CODE STRUCTURE (follow this order):
# Step 1: Inspect data structure
print("=== DATA INSPECTION ===")
print("Columns:", df.columns.tolist())
print("Date range:", df['{date_col}'].min(), "to", df['{date_col}'].max())
print("Shape:", df.shape)

# Step 2: Check if requested data exists
# Extract year/period from df and verify it matches request
# If not available, print "DATA NOT AVAILABLE" and stop

# Step 3: Filter and analyze (only if data exists)
# Your analysis code here

# Step 4: Create visualizations (if needed)
# Store figures in variables: fig, fig1, fig2, fig3, etc.
# For multiple visualizations, use: fig1 = ..., fig2 = ..., fig3 = ...

DataFrame 'df' has {len(self.data)} rows with columns: {', '.join(self.data.columns)}.
Date column: '{date_col}' (sample values: {sample_dates})

Available libraries: pd, np, go (plotly.graph_objects), px (plotly.express)
Available data: curves (dict per channel with 'slope', 'saturation', 'bottom', 'top'), hill_curve(x, slope, saturation, bottom, top) function
Available helper: to_json_safe(df) - converts Period objects to strings for plotting

CRITICAL - MULTIPLE VISUALIZATIONS:
If user asks for N visualizations, you MUST create N separate figures:
  
Example: "show curves AND contributions" → CREATE 2 FIGURES:
  fig1 = go.Figure()  # Response curves
  for channel in channels:
      fig1.add_trace(go.Scatter(...))
  
  fig2 = px.line(df, x='date', y='predicted', color='channel')  # Contributions over time

Example: "show ROI, donut, curves" → CREATE 3 FIGURES:
  fig1 = px.bar(df, x='channel', y='roi')  # ROI chart
  fig2 = px.pie(df, values='spend', names='channel', hole=0.4)  # Donut chart
  fig3 = go.Figure()  # Response curves
  
NEVER combine multiple visualizations into one figure - create separate figures for each visualization requested.
All figures (fig, fig1, fig2, fig3, etc.) will be automatically captured and displayed.

IMPORTANT: If creating quarterly/period aggregations, wrap in to_json_safe() before plotting:
  df_quarterly = df.groupby([pd.Grouper(key='date', freq='Q'), 'channel']).sum()
  df_plot = to_json_safe(df_quarterly)  # Convert Period index to strings
  fig = px.bar(df_plot, ...)

Date filtering examples:
- Convert to datetime: df['{date_col}'] = pd.to_datetime(df['{date_col}'])
- Extract year: df['year'] = df['{date_col}'].dt.year
- Extract month: df['month'] = df['{date_col}'].dt.month
- H1 (Jan-Jun): df[df['{date_col}'].dt.month <= 6]
- H2 (Jul-Dec): df[df['{date_col}'].dt.month > 6]
- Q1: df[df['{date_col}'].dt.quarter == 1]

For response curves: x = np.linspace(0, max_spend, 100); y = hill_curve(x, curves[channel]['slope'], curves[channel]['saturation'], curves[channel]['bottom'], curves[channel]['top'])

ALWAYS print() intermediate results and df.shape after filtering to verify your logic.
USE PLOTLY ONLY.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. USE PLOTLY ONLY for visualizations (go.Figure() or px functions). Store figure in 'fig' variable."
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "optimize_budget",
                    "description": f"Optimize marketing budget allocation using fitted Hill response curves. Channels available: {', '.join(channels_list)}. Uses SLSQP optimization to maximize predicted response. Extract constraints from user query (min/max per channel). Automatically generates 4 visualizations: (1) Budget allocation bar chart, (2) Donut chart of % distribution, (3) ROI comparison, (4) Response curves with allocated points marked.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "budget": {
                                "type": "number",
                                "description": "Total budget to allocate across channels (e.g., 100000000 for $100M)"
                            },
                            "num_weeks": {
                                "type": "number",
                                "description": "Planning horizon in weeks (default: 52)"
                            },
                            "min_per_channel": {
                                "type": "number",
                                "description": "Minimum budget per channel (e.g., 15000000 for $15M minimum). Extract from user query."
                            },
                            "max_per_channel": {
                                "type": "number",
                                "description": "Maximum budget per channel (e.g., 40000000 for $40M maximum). Extract from user query."
                            }
                        },
                        "required": ["budget"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information not available in the MMM data. Use this when the question requires: industry trends, emerging channels, benchmarks, best practices, external market data, or any information beyond the provided dataset.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (e.g., 'emerging marketing channels 2024', 'best performing social media platforms for advertising')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge",
                    "description": "Query the MMM knowledge base for domain expertise. Use this for: explaining MMM concepts (adstock, saturation, ROI), interpreting curve parameters, comparing to industry benchmarks, understanding best practices, getting definitions of MMM terminology. This provides expert MMM knowledge beyond the user's data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The MMM knowledge question (e.g., 'What is adstock?', 'What's a good saturation point for TV?', 'How do I interpret this slope?')"
                            }
                        },
                        "required": ["question"]
                    }
                }
            }
        ]
        
        # Build context
        summary = self._build_context()
        
        prompt = f"""You are analyzing Marketing Mix Modeling data.

Data Summary:
{summary}

User Question: {query}

IMPORTANT: If this question asks for MULTIPLE visualizations (e.g., "curves AND contributions", "ROI, donut, curves"), you MUST create MULTIPLE separate figures (fig1, fig2, fig3, etc.) - one for EACH visualization requested. Count the visualizations carefully.

If you need to analyze data or create visualizations, call execute_python with code.
Otherwise, answer directly based on the summary."""

        # Force tool usage for questions that require data analysis
        requires_tool = any(word in query.lower() for word in [
            'compare', 'vs', 'versus', 'between', 'h1', 'h2', 'quarter', 'q1', 'q2', 'q3', 'q4',
            'show', 'visualize', 'plot', 'chart', 'trend', 'performance', 'how is', 'what is',
            'allocate', 'allocation', 'budget', 'optimize'
        ])

        try:
            # Let LLM decide what to do
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an MMM Copilot with 4 tools to provide comprehensive analysis:

TOOLS AVAILABLE:
1. **execute_python** - Analyze user's data (ROI, trends, comparisons, visualizations)
2. **optimize_budget** - Allocate marketing budgets optimally
3. **query_knowledge** - Explain MMM concepts, benchmarks, best practices
4. **web_search** - Find emerging channels, industry trends

CHAIN-OF-THOUGHT REASONING (think step-by-step):
1. What is the user asking? (extract entities: channels, dates, metrics, budget, constraints)
2. What TYPE of question is this?
   - About THEIR data (ROI, performance, trends) → execute_python
   - About MMM CONCEPTS (adstock, saturation, what is X?) → query_knowledge
   - Budget allocation/optimization → optimize_budget
   - NEW channels/industry trends → web_search
3. HOW MANY visualizations requested? (count: "curves AND contributions" = 2 charts)
4. For data questions: Check if requested dates/channels exist in dataset
5. For knowledge questions: Check if it's about MMM methodology/concepts
6. For budget allocation: Extract total budget, min/max constraints
7. Execute the appropriate tool with all parameters

CRITICAL RULES:
- You MUST call a tool, never make up insights
- ALWAYS check date range FIRST: print(df[date_col].min(), df[date_col].max())
- If requested dates don't exist, say "DATA NOT AVAILABLE" - DO NOT make up numbers
- ONLY analyze data that actually exists
- Print intermediate outputs to verify your logic"""
                    },
                    {"role": "user", "content": prompt}
                ],
                tools=tools,
                tool_choice="required" if requires_tool else "auto",
                temperature=0.1,
                max_tokens=2000
            )
            
            message = response.choices[0].message
            
            # Check if tool was called
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                
                if tool_call.function.name == "optimize_budget":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    budget = args.get('budget')
                    num_weeks = args.get('num_weeks', 52)
                    min_per_channel = args.get('min_per_channel')
                    max_per_channel = args.get('max_per_channel')
                    
                    logger.info(f"Running budget optimization: ${budget:,.0f}, min=${min_per_channel}, max=${max_per_channel}")
                    
                    # Fit curves if not yet fitted
                    if not self.curves:
                        self._fit_curves()
                    
                    # Run optimization
                    from llm_copilot.core.optimization import BudgetOptimizer
                    
                    optimizer = BudgetOptimizer(
                        budget=budget,
                        channels=channels_list,
                        response_curves=self.curves,
                        num_weeks=num_weeks
                    )
                    
                    # Set constraints if provided
                    if min_per_channel or max_per_channel:
                        constraints = {}
                        for channel in channels_list:
                            constraints[channel] = {
                                'lower': min_per_channel if min_per_channel else 0,
                                'upper': max_per_channel if max_per_channel else budget
                            }
                        optimizer.set_constraints(constraints)
                        logger.info(f"Applied constraints: {constraints}")
                    
                    opt_result = optimizer.optimize()
                    
                    if not opt_result.success:
                        return {
                            'answer': f"Optimization failed: {opt_result.message}",
                            'confidence': 0.5,
                            'visualizations': [],
                            'query_type': 'error'
                        }
                    
                    # Create multiple visualizations
                    import plotly.graph_objects as go
                    import numpy as np
                    visualizations = []
                    
                    # 1. Bar chart of allocation
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=list(opt_result.allocation.keys()),
                            y=list(opt_result.allocation.values()),
                            text=[f"${v:,.0f}" for v in opt_result.allocation.values()],
                            textposition='auto'
                        )
                    ])
                    fig_bar.update_layout(
                        title=f"Optimal Budget Allocation (${budget:,.0f})",
                        xaxis_title="Channel",
                        yaxis_title="Allocated Budget ($)"
                    )
                    visualizations.append({'type': 'optimization', 'figure': fig_bar})
                    
                    # 2. Donut chart of % contribution
                    percentages = [(v / budget) * 100 for v in opt_result.allocation.values()]
                    fig_donut = go.Figure(data=[
                        go.Pie(
                            labels=list(opt_result.allocation.keys()),
                            values=percentages,
                            hole=0.4,
                            textinfo='label+percent',
                            textposition='outside'
                        )
                    ])
                    fig_donut.update_layout(title="Budget Distribution by Channel (%)")
                    visualizations.append({'type': 'donut', 'figure': fig_donut})
                    
                    # 3. ROI comparison chart
                    roi_values = []
                    for ch in opt_result.allocation.keys():
                        ch_roi = opt_result.by_channel[opt_result.by_channel['channel'] == ch]['roi'].iloc[0]
                        roi_values.append(ch_roi)
                    
                    fig_roi = go.Figure(data=[
                        go.Bar(
                            x=list(opt_result.allocation.keys()),
                            y=roi_values,
                            text=[f"{v:.2f}x" for v in roi_values],
                            textposition='auto',
                            marker_color='lightblue'
                        )
                    ])
                    fig_roi.update_layout(
                        title="Expected ROI by Channel",
                        xaxis_title="Channel",
                        yaxis_title="ROI (x)"
                    )
                    visualizations.append({'type': 'roi', 'figure': fig_roi})
                    
                    # 4. Response curves for all channels
                    if self.curves:
                        fig_curves = go.Figure()
                        for channel in opt_result.allocation.keys():
                            if channel in self.curves:
                                curve_params = self.curves[channel]
                                # Get max spend for this channel (use 2x allocated for curve visualization)
                                max_spend = opt_result.allocation[channel] * 2
                                x = np.linspace(0, max_spend, 100)
                                y = curve_params['bottom'] + (curve_params['top'] - curve_params['bottom']) * \
                                    (x**curve_params['slope']) / ((curve_params['saturation']**curve_params['slope']) + (x**curve_params['slope']))
                                
                                # Mark the allocated point
                                allocated_spend = opt_result.allocation[channel]
                                allocated_response = curve_params['bottom'] + (curve_params['top'] - curve_params['bottom']) * \
                                    (allocated_spend**curve_params['slope']) / ((curve_params['saturation']**curve_params['slope']) + (allocated_spend**curve_params['slope']))
                                
                                fig_curves.add_trace(go.Scatter(
                                    x=x, y=y,
                                    mode='lines',
                                    name=channel
                                ))
                                fig_curves.add_trace(go.Scatter(
                                    x=[allocated_spend], y=[allocated_response],
                                    mode='markers',
                                    marker=dict(size=10, symbol='star'),
                                    name=f"{channel} (allocated)",
                                    showlegend=False
                                ))
                        
                        fig_curves.update_layout(
                            title="Response Curves with Allocated Budget Points",
                            xaxis_title="Spend ($)",
                            yaxis_title="Predicted Response",
                            hovermode='x unified'
                        )
                        visualizations.append({'type': 'curves', 'figure': fig_curves})
                    
                    # Format results
                    total_roi = opt_result.predicted_response / budget
                    results_text = f"Optimal allocation for ${budget:,.0f}"
                    if min_per_channel or max_per_channel:
                        min_val = min_per_channel if min_per_channel else 0
                        max_val = max_per_channel if max_per_channel else budget
                        results_text += f" (min per channel: ${min_val:,.0f}, max per channel: ${max_val:,.0f})"
                    results_text += ":\n"
                    for ch, alloc in opt_result.allocation.items():
                        pct = (alloc / budget) * 100
                        ch_roi = opt_result.by_channel[opt_result.by_channel['channel'] == ch]['roi'].iloc[0]
                        results_text += f"  {ch}: ${alloc:,.0f} ({pct:.1f}%) - ROI: {ch_roi:.2f}x\n"
                    results_text += f"Total ROI: {total_roi:.2f}x"
                    
                    # Generate answer
                    final_response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": """You are a Marketing Mix Modeling optimization expert providing executive-ready recommendations.

ANSWER STRUCTURE:
1. **Recommended Allocation**: Present the optimized budget breakdown by channel with actual dollar amounts and percentages
2. **ROI Analysis**: Explain the expected ROI for each channel and total portfolio
3. **Rationale**: Explain WHY this allocation is optimal (which channels are efficient, saturation points, diminishing returns)
4. **Constraints Met**: Confirm all constraints were satisfied (min/max per channel)
5. **Key Trade-offs**: Highlight any important decisions made in the optimization

FORMATTING RULES (CRITICAL FOR READABILITY):
- For currency: Write as "42.21 million dollars" or "21.26M dollars" (avoid inline $ symbols)
- For ROI values: "ROI of 3.37" (not "$3.37")
- ALWAYS add space after periods: "3.37. This is..." NOT "3.37.This is..."
- ALWAYS add space after commas: "allocation, considering..." NOT "allocation,considering..."
- Use bullet points (- or •) with proper indentation
- Add blank lines between major sections
- Use **bold** for section headers

CRITICAL - USE ACTUAL NAMES FROM DATA:
- Use ACTUAL channel names from the optimization results (e.g., "TV", "Search", "Social", not "Channel A", "Channel B")
- Use the EXACT dollar amounts from the results
- Use the EXACT ROI figures from the results
- DO NOT anonymize, genericize, or replace real names with placeholders

TONE: Confident, data-driven, actionable. This is a recommendation to executives."""},
                            {"role": "user", "content": f"""Question: {query}

Optimization Results:
{results_text}

Provide a comprehensive, executive-ready recommendation explaining:
- The optimal allocation and why it's optimal
- Expected ROI and performance
- How constraints were met
- Key strategic insights from the optimization

IMPORTANT:
- Use ACTUAL channel names from the results (TV, Search, Social, Display, Radio - NOT "Channel 1", "Channel A", etc.)
- Use EXACT dollar amounts and ROI figures from the optimization results
- Ensure proper spacing (space after periods and commas)
- Avoid inline $ symbols (write "X million dollars" instead of "$X million")"""}
                        ],
                        temperature=0.3,
                        max_tokens=1200
                    )
                    
                    return {
                        'answer': final_response.choices[0].message.content,
                        'confidence': 0.95,
                        'visualizations': visualizations,
                        'query_type': 'optimization'
                    }
                
                elif tool_call.function.name == "execute_python":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    code = args.get('code', '')
                    
                    logger.info(f"Executing code:\n{code}")
                    
                    # Execute code (with retry on error)
                    result = self._execute_code(code)
                    
                    if result['error']:
                        # Retry once with error feedback
                        logger.warning(f"Code failed, retrying with error feedback: {result['error']}")
                        retry_response = self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": """You analyze MMM data by using tools. For data analysis/visualizations: call execute_python tool.

CHAIN-OF-THOUGHT: 
1. Check what went wrong in the error
2. Check data structure (columns, dtypes, date range)
3. Verify requested data exists
4. Fix the code
5. Always inspect data first before analysis"""},
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                                {"role": "tool", "tool_call_id": tool_call.id, "content": f"ERROR: {result['error']}\n\nThink step-by-step and fix:\n1. What caused this error?\n2. Check: print(df['{date_col}'].min(), df['{date_col}'].max()) - does requested date exist?\n3. Check: print(df.columns) - are column names correct?\n4. Fix the code with proper data validation"},
                            ],
                            tools=tools,
                            tool_choice="required",
                            temperature=0.1,
                            max_tokens=2000
                        )
                        
                        retry_message = retry_response.choices[0].message
                        if retry_message.tool_calls:
                            retry_tool_call = retry_message.tool_calls[0]
                            if retry_tool_call.function.name == "execute_python":
                                retry_args = json.loads(retry_tool_call.function.arguments)
                                retry_code = retry_args.get('code', '')
                                logger.info(f"Retrying with fixed code:\n{retry_code}")
                                result = self._execute_code(retry_code)
                                code = retry_code
                        
                        # If still error after retry, return error
                        if result['error']:
                            return {
                                'answer': f"Code execution failed: {result['error']}",
                                'confidence': 0.5,
                                'visualizations': [],
                                'query_type': 'error'
                            }
                    
                    # Generate final answer
                    final_response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": """You are an expert Marketing Mix Modeling analyst providing executive-friendly insights.

ANSWER STRUCTURE:
1. **Key Findings**: Start with 2-3 headline insights (actual numbers from data)
2. **Detailed Analysis**: Break down the data with specific metrics, comparisons, trends
3. **Context**: Compare to benchmarks if available, explain what the numbers mean
4. **Actionable Recommendations**: Specific next steps based on the data

FORMATTING RULES (CRITICAL FOR READABILITY):
- For currency: Write as "4.99 million dollars" or "10.60M dollars" (avoid inline $ symbols)
- For ROI values: "ROI of 2.12" (not "$2.12")
- ALWAYS add space after periods: "2.12. This is..." NOT "2.12.This is..."
- ALWAYS add space after commas: "spend, resulting in..." NOT "spend,resulting in..."
- Use bullet points (- or •) with proper indentation
- Add blank lines between major sections
- Use **bold** for section headers

CRITICAL - USE ACTUAL NAMES FROM DATA:
- Use ACTUAL channel names from the data (e.g., "TV", "Search", "Social", not "Channel A", "Channel B")
- Use ACTUAL column names from the data (e.g., "spend", "predicted", not generic terms)
- Use ACTUAL date ranges from the data (e.g., "2024-01-01 to 2024-12-31", not "the year")
- DO NOT anonymize, genericize, or replace real names with placeholders

TONE: Professional, data-driven, actionable. Use specific numbers from the analysis results."""},
                            {"role": "user", "content": f"""Question: {query}

Analysis Results (from code execution):
{result['output']}

Provide a comprehensive, executive-friendly answer with:
- Specific numbers and metrics from the results
- Comparisons and context
- Trends and patterns observed
- Actionable insights and recommendations

IMPORTANT: 
- Use ACTUAL channel names from the data (TV, Search, Social, Display, Radio - NOT "Channel A", "Channel B", etc.)
- Use EXACT numbers and dates from the analysis results
- Ensure proper spacing (space after periods and commas)
- Avoid inline $ symbols (write "X million dollars" instead of "$X million")"""}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                    
                    visualizations = []
                    # Add all captured figures
                    for idx, fig in enumerate(result.get('figures', [])):
                        visualizations.append({'type': 'analysis', 'figure': fig})
                    
                    logger.info(f"Returning {len(visualizations)} visualizations to UI")
                    
                    return {
                        'answer': final_response.choices[0].message.content,
                        'confidence': 0.9,
                        'visualizations': visualizations,
                        'query_type': 'code_generated',
                        'code': code
                    }
                
                elif tool_call.function.name == "web_search":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    search_query = args.get('query', '')
                    
                    logger.info(f"Performing web search: {search_query}")
                    
                    # Use LLM's built-in knowledge for marketing trends
                    # (Web search API integration would require additional setup and API keys)
                    final_response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": """You are a marketing expert with knowledge of current industry trends, emerging channels, and best practices.

PROVIDE INSIGHTS ON:
- Emerging marketing channels (TikTok, YouTube Shorts, Podcasts, Connected TV, etc.)
- Industry trends and benchmarks
- Channel recommendations based on target audience
- Performance expectations for different channels

FORMAT:
- Use actual channel names
- Provide specific, actionable recommendations
- Include ROI expectations if available
- Mention pros/cons of each recommended channel"""},
                            {"role": "user", "content": f"""Question: {query}

Current MMM Data Context:
{summary}

Current channels: TV, Search, Social, Display, Radio

Provide comprehensive recommendations for new channels to expand into, considering:
- Current performance of existing channels
- Industry trends and emerging platforms
- Expected ROI and investment levels
- Target audience considerations"""}
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    return {
                        'answer': f"**Based on industry trends and your current channel mix:**\n\n{final_response.choices[0].message.content}",
                        'confidence': 0.75,
                        'visualizations': [],
                        'query_type': 'web_search'
                    }
                
                elif tool_call.function.name == "query_knowledge":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    knowledge_question = args.get('question', '')
                    
                    logger.info(f"Querying knowledge base: {knowledge_question}")
                    
                    # Search knowledge base for relevant information (now includes glossary + best practices via semantic search)
                    kb_results = self.knowledge_base.search(knowledge_question, top_k=5)
                    
                    # Build knowledge context from semantic search results
                    knowledge_context = "**Knowledge Base Results (Semantic Search):**\n"
                    for idx, doc in enumerate(kb_results, 1):
                        doc_type = doc.get('metadata', {}).get('type', 'unknown')
                        similarity = doc.get('similarity', 0)
                        knowledge_context += f"\n{idx}. [{doc_type.upper()}] (similarity: {similarity:.2f})\n{doc['content']}\n"
                    
                    # Generate expert answer
                    final_response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": """You are an expert MMM consultant explaining concepts to executives.

RULES:
- Use the knowledge base results provided (retrieved via semantic search)
- Knowledge base includes: glossary definitions, industry benchmarks, best practices, and fitted curves
- Explain concepts clearly with examples
- Relate to the user's data when relevant
- Include industry benchmarks and citations when available
- Be educational but concise
- Format with bullet points for readability"""},
                            {"role": "user", "content": f"""Question: {knowledge_question}

Knowledge Base Information:
{knowledge_context}

User's Data Context:
{summary}

Provide a clear, educational answer that helps the user understand this MMM concept."""}
                        ],
                        temperature=0.5,
                        max_tokens=1200
                    )
                    
                    return {
                        'answer': final_response.choices[0].message.content,
                        'confidence': 0.9,
                        'visualizations': [],
                        'query_type': 'knowledge'
                    }
            
            # Direct answer (no code needed)
            # BUT: if the query required a tool and LLM didn't use one, reject it
            if requires_tool:
                logger.error(f"LLM failed to use tool for query requiring data analysis: {query}")
                return {
                    'answer': "ERROR: This question requires analyzing actual data, but the AI failed to execute code. The response would be generic/hallucinated. Please try rephrasing your question or contact support.",
                    'confidence': 0.0,
                    'visualizations': [],
                    'query_type': 'error'
                }
            
            return {
                'answer': message.content,
                'confidence': 0.85,
                'visualizations': [],
                'query_type': 'descriptive'
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'confidence': 0.0,
                'visualizations': [],
                'query_type': 'error'
            }
    
    def _build_context(self) -> str:
        """Build data summary for context"""
        lines = [f"Dataset: {len(self.data)} rows × {len(self.data.columns)} columns"]
        lines.append(f"Columns: {', '.join(self.data.columns)}")
        
        # If there's a channel column, provide channel-level summary
        if 'channel' in self.data.columns:
            lines.append(f"\nChannels: {', '.join(self.data['channel'].unique())}")
            
            # Calculate ROI per channel
            for channel in self.data['channel'].unique():
                ch_data = self.data[self.data['channel'] == channel]
                if 'spend' in ch_data.columns and 'predicted' in ch_data.columns:
                    total_spend = ch_data['spend'].sum()
                    total_pred = ch_data['predicted'].sum()
                    roi = total_pred / total_spend if total_spend > 0 else 0
                    lines.append(f"  {channel}: Spend ${total_spend:,.0f}, Predicted ${total_pred:,.0f}, ROI {roi:.2f}x")
        
        return "\n".join(lines)
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code in sandboxed environment"""
        import sys
        from io import StringIO
        import traceback
        import re
        
        # Remove problematic calls
        code = re.sub(r'fig\.show\(\s*\)', '# fig.show() suppressed', code, flags=re.IGNORECASE)
        code = re.sub(r'fig\.write_html\([^)]+\)', '# fig.write_html() suppressed', code, flags=re.IGNORECASE)
        
        # Validate no matplotlib
        if 'matplotlib' in code.lower():
            return {'output': '', 'figure': None, 'error': 'matplotlib not available. Use plotly only.'}
        
        # Setup namespace
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Define hill_curve function
        def hill_curve(x, slope, saturation, bottom, top):
            """Hill transformation for response curves"""
            return bottom + (top - bottom) * (x**slope) / ((saturation**slope) + (x**slope))
        
        # Prepare data - convert any Period objects to strings for JSON serialization
        df_copy = self.data.copy()
        
        # Convert Period columns
        for col in df_copy.columns:
            if hasattr(df_copy[col].dtype, 'name') and 'period' in df_copy[col].dtype.name.lower():
                df_copy[col] = df_copy[col].astype(str)
        
        # Also provide a helper function to convert periods in the namespace
        def to_json_safe(df_input):
            """Convert DataFrame with Period objects to JSON-safe format"""
            df_safe = df_input.copy()
            for col in df_safe.columns:
                if hasattr(df_safe[col].dtype, 'name') and 'period' in df_safe[col].dtype.name.lower():
                    df_safe[col] = df_safe[col].astype(str)
                # Also handle Period index
                if hasattr(df_safe.index, 'dtype') and hasattr(df_safe.index.dtype, 'name') and 'period' in df_safe.index.dtype.name.lower():
                    df_safe.index = df_safe.index.astype(str)
            return df_safe
        
        namespace = {
            'df': df_copy,
            'pd': pd,
            'np': np,
            'go': go,
            'px': px,
            'curves': self.curves.copy() if self.curves else {},
            'hill_curve': hill_curve,
            'to_json_safe': to_json_safe,
            'fig': None
        }
        
        # Capture output
        stdout_capture = StringIO()
        original_stdout = sys.stdout
        sys.stdout = stdout_capture
        
        try:
            exec(code, namespace)
            sys.stdout = original_stdout
            
            output = stdout_capture.getvalue() or 'Code executed successfully'
            
            # Capture ALL Plotly figures from namespace (fig, fig1, fig2, fig3, etc.)
            figures = []
            
            # Define helper function for Period conversion (reusable)
            def convert_figure_to_json_safe(figure):
                """Convert a Plotly figure to JSON-safe format by replacing Period objects"""
                if not figure or not hasattr(figure, 'to_html'):
                    return None
                
                try:
                    # First, try to convert the figure to JSON to see if it works
                    test_json = figure.to_json()
                    # If it works, great! Return as-is
                    return figure
                except Exception as json_error:
                    # If JSON serialization fails (likely due to Period), fix it
                    logger.info(f"JSON serialization failed, converting Periods: {json_error}")
                    try:
                        fig_dict = figure.to_dict()
                        
                        # Recursively convert all non-JSON-serializable objects to strings
                        def make_json_safe(obj):
                            if isinstance(obj, dict):
                                return {k: make_json_safe(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [make_json_safe(item) for item in obj]
                            elif isinstance(obj, (str, int, float, bool, type(None))):
                                return obj
                            # Try to serialize it
                            else:
                                try:
                                    import json
                                    json.dumps(obj)
                                    return obj
                                except (TypeError, ValueError):
                                    # Not JSON serializable - convert to string
                                    return str(obj)
                        
                        fig_dict_clean = make_json_safe(fig_dict)
                        
                        # Recreate figure from cleaned dict
                        import plotly.graph_objects as go
                        figure_clean = go.Figure(fig_dict_clean)
                        logger.info("Successfully converted figure to JSON-safe format")
                        return figure_clean
                        
                    except Exception as conversion_error:
                        logger.error(f"Failed to convert figure: {conversion_error}")
                        return None
            
            # Check for 'fig' variable
            if 'fig' in namespace:
                fig = convert_figure_to_json_safe(namespace['fig'])
                if fig:
                    figures.append(fig)
                    logger.info("Captured figure: fig")
            
            # Check for fig1, fig2, fig3, ... fig10
            for i in range(1, 11):
                fig_name = f'fig{i}'
                if fig_name in namespace:
                    fig = convert_figure_to_json_safe(namespace[fig_name])
                    if fig:
                        figures.append(fig)
                        logger.info(f"Captured figure: {fig_name}")
            
            logger.info(f"Total figures captured: {len(figures)}")
            
            return {
                'output': output,
                'figures': figures,
                'error': None
            }
            
        except Exception as e:
            sys.stdout = original_stdout
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"Execution error: {error_msg}")
            
            return {
                'output': '',
                'figures': [],
                'error': error_msg
            }
    
    def _fit_curves(self):
        """Fit Hill response curves for all channels and populate knowledge base"""
        from llm_copilot.core.response_curves import ResponseCurveGenerator
        
        logger.info("Fitting response curves...")
        curve_gen = ResponseCurveGenerator(self.data, date_col=self.date_col)
        
        if 'channel' in self.data.columns:
            channels = self.data['channel'].unique().tolist()
            self.curves = curve_gen.fit_curves(channels=channels)
            logger.info(f"Fitted curves for {len(self.curves)} channels")
            
            # Populate Knowledge Base with curve results for RAG
            logger.info("Populating Knowledge Base with curve results...")
            self.knowledge_base.add_from_curves(self.curves)
            
            # Add channel benchmarks for comparison
            for channel in channels:
                channel_upper = channel.upper()
                if channel_upper in self.expertise.CHANNEL_BENCHMARKS:
                    benchmark = self.expertise.CHANNEL_BENCHMARKS[channel_upper]
                    content = f"""Industry Benchmark for {channel}:
Typical ROI Range: {benchmark['typical_roi_range']}
Typical Saturation: {benchmark['typical_saturation']}
Adstock Window: {benchmark['adstock_window']}
Best For: {benchmark['best_for']}
Seasonality: {benchmark['seasonality']}"""
                    self.knowledge_base.add_document(
                        doc_id=f'benchmark_{channel}',
                        content=content,
                        metadata={'type': 'benchmark', 'channel': channel}
                    )
            
            # Add glossary to KnowledgeBase for semantic search
            logger.info("Adding glossary to Knowledge Base...")
            for term, info in self.expertise.MMM_GLOSSARY.items():
                content = f"""{term.upper()}

Definition: {info['definition']}
Formula: {info.get('formula', 'N/A')}
Context: {info['context']}
Synonyms: {', '.join(info.get('synonyms', []))}"""
                
                self.knowledge_base.add_document(
                    doc_id=f'glossary_{term}',
                    content=content,
                    metadata={'type': 'glossary', 'term': term}
                )
            
            logger.info(f"Added {len(self.expertise.MMM_GLOSSARY)} glossary terms to Knowledge Base")
            
            # Add best practices to KnowledgeBase for semantic search
            logger.info("Adding best practices to Knowledge Base...")
            for practice_name, practice_info in self.expertise.BEST_PRACTICES.items():
                content = f"""Best Practice: {practice_name.replace('_', ' ').title()}

Rule: {practice_info['rule']}
Action: {practice_info['action']}
Citation: {practice_info['citation']}"""
                
                self.knowledge_base.add_document(
                    doc_id=f'practice_{practice_name}',
                    content=content,
                    metadata={'type': 'best_practice', 'practice': practice_name}
                )
            
            logger.info(f"Added {len(self.expertise.BEST_PRACTICES)} best practices to Knowledge Base")
            
            logger.info("Knowledge Base fully populated with curves, benchmarks, glossary, and best practices")
        else:
            logger.warning("No 'channel' column found, cannot fit curves")

