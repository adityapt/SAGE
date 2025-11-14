"""
Agentic Planning Module

Inspired by DeepAnalyze's agentic capabilities for autonomous analysis.
Automatically plans and executes multi-step MMM analysis workflows.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class AnalysisStep:
    """Single step in analysis plan"""
    step_id: int
    action: str  # 'load_data', 'generate_curves', 'optimize', 'visualize', 'analyze'
    description: str
    parameters: Dict
    dependencies: List[int]
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    result: Optional[Dict] = None


class AgenticPlanner:
    """
    Autonomous planner for MMM analysis workflows.
    
    Inspired by DeepAnalyze's agentic approach:
    - Automatically decomposes complex queries into analysis steps
    - Creates executable plans with dependencies
    - Self-corrects when steps fail
    - Generates comprehensive reports
    
    Parameters
    ----------
    api_key : str
        OpenAI API key
    model : str, default="gpt-4o"
        LLM for planning
        
    Examples
    --------
    >>> planner = AgenticPlanner(api_key="...")
    >>> plan = planner.create_plan(
    ...     "Analyze TV performance, compare to Search, and recommend optimal allocation"
    ... )
    >>> # Plan automatically broken into: load data → curves → comparison → optimization → report
    """
    
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o",
        temperature: float = 0.1
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
        logger.info("Initialized AgenticPlanner")
    
    def create_plan(
        self,
        query: str,
        *,
        available_data: Optional[List[str]] = None,
        context: Optional[Dict] = None
    ) -> List[AnalysisStep]:
        """
        Create autonomous analysis plan from user query.
        
        Parameters
        ----------
        query : str
            User's analysis request
        available_data : List[str], optional
            Available data sources
        context : Dict, optional
            Additional context (curves generated, optimization run, etc.)
            
        Returns
        -------
        List[AnalysisStep]
            Ordered list of analysis steps with dependencies
        """
        logger.info(f"Creating analysis plan for: {query}")
        
        # Build planning prompt
        system_prompt = self._get_planning_system_prompt()
        user_prompt = self._build_planning_prompt(query, available_data, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            plan_text = response.choices[0].message.content
            plan = self._parse_plan(plan_text)
            
            logger.info(f"Created plan with {len(plan)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            # Fallback to simple plan
            return self._create_fallback_plan(query)
    
    def _get_planning_system_prompt(self) -> str:
        """System prompt for plan generation"""
        return """You are an autonomous MMM analysis planner inspired by DeepAnalyze.

Your task: Break down user requests into executable Python code steps.

Available Actions:
1. EXECUTE_CODE - Write and run Python code for ANY analysis/visualization
2. GENERATE_CURVES - Fit response curves using Hill transformation (uses existing tool)
3. OPTIMIZE - Run budget optimization with constraints (uses existing tool)
4. REPORT - Generate comprehensive analysis report

**PREFER EXECUTE_CODE for flexibility** - you can write pandas/plotly/numpy code to do anything:
- Filter data by date ranges
- Create custom visualizations
- Aggregate metrics
- Compare channels or time periods
- Statistical analysis

Output Format (JSON):
```json
{
  "steps": [
    {
      "step_id": 1,
      "action": "EXECUTE_CODE",
      "description": "Filter Q3 2024 data for TV and Search",
      "parameters": {
        "code": "q3_data = df[(df['date'] >= '2024-07-01') & (df['date'] <= '2024-09-30') & df['channel'].isin(['TV', 'Search'])]; result = q3_data.groupby('channel')[['spend', 'predicted']].sum(); print(result)"
      },
      "dependencies": []
    },
    {
      "step_id": 2,
      "action": "EXECUTE_CODE",
      "description": "Create combined response curve visualization",
      "parameters": {
        "code": "fig = go.Figure(); for ch in ['TV', 'Search']: ch_data = df[df['channel']==ch]; fig.add_trace(go.Scatter(x=ch_data['spend'], y=ch_data['predicted'], mode='markers', name=ch)); fig.write_html('workspace/combined_curve.html'); print('Saved')"
      },
      "dependencies": [1]
    },
    {
      "step_id": 3,
      "action": "REPORT",
      "description": "Generate comprehensive analysis report",
      "parameters": {},
      "dependencies": [1, 2]
    }
  ],
  "reasoning": "Filter relevant data, create visualization, generate report"
}
```

Rules:
- **Write actual Python code** - don't use generic placeholders
- Available in code: df (data), pd, np, go (plotly), px, date_col, channel_col
- Keep steps focused (one task per step)
- Respect dependencies
- End with REPORT for comprehensive analysis
- Be creative - you can write ANY valid Python code"""
    
    def _build_planning_prompt(
        self,
        query: str,
        available_data: Optional[List[str]],
        context: Optional[Dict]
    ) -> str:
        """Build user prompt with context"""
        prompt = f"User Request: {query}\n\n"
        
        if available_data:
            prompt += f"Available Data: {', '.join(available_data)}\n"
        
        if context:
            prompt += "Current Context:\n"
            if context.get('curves_generated'):
                prompt += "- Response curves: Already generated\n"
            if context.get('optimization_run'):
                prompt += "- Optimization: Already completed\n"
            if context.get('available_metrics'):
                prompt += f"- Available metrics: {', '.join(context['available_metrics'])}\n"
        
        prompt += "\nGenerate analysis plan (JSON format):"
        return prompt
    
    def _parse_plan(self, plan_text: str) -> List[AnalysisStep]:
        """Parse LLM response into AnalysisStep objects"""
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'```json\n(.*?)\n```', plan_text, re.DOTALL)
        if json_match:
            plan_json = json.loads(json_match.group(1))
        else:
            # Try parsing entire response as JSON
            plan_json = json.loads(plan_text)
        
        steps = []
        for step_data in plan_json.get('steps', []):
            step = AnalysisStep(
                step_id=step_data['step_id'],
                action=step_data['action'],
                description=step_data['description'],
                parameters=step_data.get('parameters', {}),
                dependencies=step_data.get('dependencies', [])
            )
            steps.append(step)
        
        return steps
    
    def _create_fallback_plan(self, query: str) -> List[AnalysisStep]:
        """Fallback plan when LLM planning fails"""
        q_lower = query.lower()
        
        steps = []
        step_id = 1
        
        # Always start with data validation
        steps.append(AnalysisStep(
            step_id=step_id,
            action='LOAD_DATA',
            description='Validate MMM data',
            parameters={},
            dependencies=[]
        ))
        step_id += 1
        
        # Add steps based on keywords
        if any(kw in q_lower for kw in ['curve', 'saturation', 'response']):
            steps.append(AnalysisStep(
                step_id=step_id,
                action='GENERATE_CURVES',
                description='Generate response curves',
                parameters={'channels': 'all'},
                dependencies=[1]
            ))
            step_id += 1
        
        if any(kw in q_lower for kw in ['optimal', 'optimize', 'allocation']):
            steps.append(AnalysisStep(
                step_id=step_id,
                action='OPTIMIZE',
                description='Run budget optimization',
                parameters={},
                dependencies=[step_id - 1]
            ))
            step_id += 1
        
        if any(kw in q_lower for kw in ['visualiz', 'plot', 'chart', 'show']):
            steps.append(AnalysisStep(
                step_id=step_id,
                action='VISUALIZE',
                description='Create visualizations',
                parameters={'type': 'auto'},
                dependencies=[step_id - 1]
            ))
            step_id += 1
        
        # Always end with report
        steps.append(AnalysisStep(
            step_id=step_id,
            action='REPORT',
            description='Generate analysis report',
            parameters={},
            dependencies=[step_id - 1]
        ))
        
        logger.info(f"Created fallback plan with {len(steps)} steps")
        return steps
    
    def can_execute_step(self, step: AnalysisStep, completed_steps: List[int]) -> bool:
        """Check if step dependencies are satisfied"""
        return all(dep in completed_steps for dep in step.dependencies)
    
    def update_step_status(
        self,
        step: AnalysisStep,
        status: str,
        result: Optional[Dict] = None
    ) -> None:
        """Update step execution status"""
        step.status = status
        if result:
            step.result = result
        
        logger.info(f"Step {step.step_id} ({step.action}): {status}")


class ReportGenerator:
    """
    Generate comprehensive analysis reports.
    
    Inspired by DeepAnalyze's report generation capabilities.
    Creates analyst-grade markdown reports with findings, visualizations, and recommendations.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_report(
        self,
        query: str,
        analysis_results: Dict,
        *,
        include_technical_details: bool = False
    ) -> str:
        """
        Generate comprehensive MMM analysis report.
        
        Parameters
        ----------
        query : str
            Original analysis question
        analysis_results : Dict
            Results from all analysis steps
        include_technical_details : bool, default=False
            Whether to include technical methodology details
            
        Returns
        -------
        str
            Markdown-formatted analysis report
        """
        logger.info("Generating comprehensive report")
        
        system_prompt = """You are an expert MMM analyst generating comprehensive reports.

Structure your report with:
1. Executive Summary
2. Key Findings (bullet points, quantitative)
3. Detailed Analysis (with specific numbers and comparisons)
4. Visualizations (reference charts)
5. Recommendations (actionable, prioritized)
6. Caveats and Assumptions

Use markdown formatting. Be specific with numbers. Provide context for metrics."""

        user_prompt = f"""Generate a comprehensive MMM analysis report.

Original Question: {query}

Analysis Results:
{self._format_results(analysis_results)}

Create an analyst-grade report with specific insights, numbers, and actionable recommendations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Slightly higher for natural report writing
                max_tokens=2000
            )
            
            report = response.choices[0].message.content
            
            if include_technical_details:
                report += "\n\n" + self._add_technical_appendix(analysis_results)
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return self._generate_fallback_report(query, analysis_results)
    
    def _format_results(self, results: Dict) -> str:
        """Format analysis results for prompt"""
        formatted = []
        
        for step_name, step_result in results.items():
            formatted.append(f"**{step_name}**:")
            
            if isinstance(step_result, dict):
                for key, value in step_result.items():
                    formatted.append(f"  - {key}: {value}")
            else:
                formatted.append(f"  {step_result}")
        
        return "\n".join(formatted)
    
    def _add_technical_appendix(self, results: Dict) -> str:
        """Add technical methodology details"""
        appendix = "## Technical Appendix\n\n"
        appendix += "### Methodology\n"
        appendix += "- **Response Curves**: Hill transformation with scipy curve_fit\n"
        appendix += "- **Optimization**: trust-constr algorithm (interior-point method)\n"
        appendix += "- **Confidence Intervals**: Bootstrap resampling (if applicable)\n\n"
        
        if 'curves' in results:
            appendix += "### Curve Parameters\n"
            for channel, params in results['curves'].items():
                appendix += f"- **{channel}**: Slope={params.get('slope', 'N/A'):.2f}, "
                appendix += f"Saturation=${params.get('saturation', 0):,.0f}, R²={params.get('r_2', 0):.3f}\n"
        
        return appendix
    
    def _generate_fallback_report(self, query: str, results: Dict) -> str:
        """Simple fallback report"""
        report = f"# MMM Analysis Report\n\n"
        report += f"**Question**: {query}\n\n"
        report += f"## Results\n\n"
        
        for step_name, step_result in results.items():
            report += f"### {step_name}\n{step_result}\n\n"
        
        return report

