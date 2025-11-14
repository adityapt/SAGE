"""
Code generation module for MMM analysis (DeepAnalyze-inspired).

This module enables the LLM to write and execute Python code for custom analysis.
"""

import logging
import sys
from io import StringIO
from typing import Dict, Any, Optional
import traceback
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generate and execute Python code for MMM analysis."""
    
    def __init__(self, data: pd.DataFrame, curves: Dict[str, Dict], api_key: str, base_url: str = None):
        """
        Initialize code generator.
        
        Parameters
        ----------
        data : pd.DataFrame
            MMM data with columns: date, channel, spend, predicted
        curves : Dict[str, Dict]
            Fitted response curves: {channel: {slope, saturation, bottom, top, r_2}}
        api_key : str
            OpenAI API key
        base_url : str, optional
            OpenAI base URL
        """
        self.data = data
        self.curves = curves
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        
    def generate_and_execute(self, query: str) -> Dict[str, Any]:
        """
        Generate Python code for the query and execute it.
        
        Parameters
        ----------
        query : str
            User's analysis request
            
        Returns
        -------
        Dict
            - code: Generated Python code
            - output: Text output from execution
            - visualization: Path to generated visualization (if any)
            - error: Error message (if execution failed)
        """
        # Step 1: Generate code
        code = self._generate_code(query)
        
        if not code:
            return {'error': 'Failed to generate code', 'code': None, 'output': None}
        
        logger.info(f"Generated code ({len(code)} chars)")
        
        # Step 2: Execute code
        result = self._execute_code(code)
        result['code'] = code
        
        return result
    
    def _generate_code(self, query: str) -> Optional[str]:
        """Generate Python code using LLM."""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # Build context
        context = f"""You are a Python code generator for Marketing Mix Model (MMM) analysis.

AVAILABLE DATA:
- df: DataFrame with {len(self.data)} rows
  Columns: date (datetime), channel (str), spend (float), impressions (float), predicted (float)
  Date range: {self.data['date'].min()} to {self.data['date'].max()}
  Channels: {', '.join(self.data['channel'].unique())}

- curves: Dict with fitted Hill curve parameters
  Keys: {', '.join(self.curves.keys())}
  Structure: {{'slope': float, 'saturation': float, 'bottom': float, 'top': float, 'r_2': float}}

AVAILABLE FUNCTIONS:
- hill_curve(x, slope, saturation, bottom, top): Generate Hill curve points
- Libraries: pandas (pd), numpy (np), plotly.graph_objects (go), plotly.express (px)

USER REQUEST: {query}

INSTRUCTIONS:
1. Write Python code to answer the user's request
2. For data analysis (filtering, aggregation, comparison):
   - Query the actual data using pandas
   - Filter by date ranges (e.g., H1 = Jan-Jun, H2 = Jul-Dec)
   - Calculate exact numbers (don't assume or estimate)
   - Print clear, specific results with numbers
3. For visualization:
   - Use plotly (go or px)
   - Create the figure and STORE IT in variable 'fig' (REQUIRED)
   - Do NOT call fig.show() or fig.write_html()
4. Always use actual data, never make assumptions
5. Print results clearly with context

CRITICAL: Do NOT use fig.show() or fig.write_html()

OUTPUT ONLY EXECUTABLE PYTHON CODE (no markdown, no explanation):"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Python code generator. Output ONLY executable Python code, no markdown blocks, no explanations."},
                    {"role": "user", "content": context}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            code = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if code.startswith('```'):
                lines = code.split('\n')
                # Remove first line (```python or ```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                code = '\n'.join(lines)
            
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code in a sandboxed environment."""
        import os
        import re
        os.makedirs("workspace", exist_ok=True)
        
        # Log the code being executed
        logger.info(f"Executing code:\n{code}")
        
        # Validate: No matplotlib imports
        if 'matplotlib' in code.lower() or 'import plt' in code.lower():
            return {
                'output': '',
                'figure': None,
                'visualization': None,
                'error': "Error: matplotlib is not available. Please use plotly instead (go.Figure() or px functions)."
            }
        
        # Remove fig.show() calls to prevent new browser windows
        code = re.sub(r'fig\.show\(\s*\)', '# fig.show() suppressed', code, flags=re.IGNORECASE)
        code = re.sub(r'fig\.write_html\([^)]+\)', '# fig.write_html() suppressed', code, flags=re.IGNORECASE)
        
        # Define hill_curve function
        def hill_curve(x, slope, saturation, bottom, top):
            """Hill transformation function."""
            return bottom + (top - bottom) * (x**slope) / ((saturation**slope) + (x**slope))
        
        # Verify DataFrame has expected columns
        logger.info(f"DataFrame columns: {list(self.data.columns)}")
        logger.info(f"DataFrame shape: {self.data.shape}")
        
        # Create execution namespace
        namespace = {
            'df': self.data.copy(),
            'curves': self.curves.copy(),
            'pd': pd,
            'np': np,
            'go': go,
            'px': px,
            'hill_curve': hill_curve,
            'date_col': 'date',
            'channel_col': 'channel',
            'fig': None  # Will capture the figure here
        }
        
        # Capture stdout
        stdout_capture = StringIO()
        original_stdout = sys.stdout
        sys.stdout = stdout_capture
        
        try:
            # Execute code
            exec(code, namespace)
            
            # Restore stdout
            sys.stdout = original_stdout
            output = stdout_capture.getvalue()
            
            # Extract Plotly figure from namespace (if created)
            figure = namespace.get('fig')
            
            if figure is not None and hasattr(figure, 'to_html'):
                logger.info("Captured Plotly figure object")
            else:
                figure = None
            
            return {
                'output': output or 'Code executed successfully',
                'figure': figure,  # Direct Plotly object
                'visualization': None,  # Deprecated: use 'figure' instead
                'error': None
            }
            
        except Exception as e:
            # Restore stdout
            sys.stdout = original_stdout
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            return {
                'output': stdout_capture.getvalue(),
                'figure': None,
                'visualization': None,
                'error': error_msg
            }

