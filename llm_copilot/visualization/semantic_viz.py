"""
Semantic Visualization Layer

Automatically generates appropriate visualizations based on user questions and MMM data.
"""

from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI

from llm_copilot.visualization.metrics import MetricDefinitions

logger = logging.getLogger(__name__)


class SemanticVisualizer:
    """
    Automatically generate visualizations based on natural language questions.
    
    Supports all MMM metrics and outputs:
    - Contribution over time
    - ROI/ROAS by channel
    - Response curves and saturation
    - Attribution/decomposition
    - Optimization results
    - Comparative analysis
    
    Parameters
    ----------
    data : pd.DataFrame
        MMM data with columns: week_monday, channel, spend, predicted, etc.
    api_key : str, optional
        OpenAI API key for semantic understanding (if not using keyword matching)
        
    Examples
    --------
    >>> viz = SemanticVisualizer(data=mmm_data)
    >>> fig = viz.visualize("Show me ROI by channel")
    >>> fig.show()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        api_key: Optional[str] = None
    ):
        self.data = data.copy()
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.metrics = MetricDefinitions()
        
        logger.info(f"Initialized SemanticVisualizer with {len(data)} rows")
    
    def visualize(
        self,
        question: str,
        *,
        curves: Optional[Dict] = None,
        optimization_result: Optional[pd.DataFrame] = None
    ) -> Tuple[go.Figure, str]:
        """
        Generate visualization based on natural language question.
        
        Parameters
        ----------
        question : str
            User question (e.g., "Show me ROI by channel")
        curves : Dict, optional
            Response curves if available
        optimization_result : pd.DataFrame, optional
            Optimization results if available
            
        Returns
        -------
        Tuple[go.Figure, str]
            (Plotly figure, explanation text with metric definitions)
        """
        logger.info(f"Generating visualization for: {question}")
        
        # Determine visualization type
        viz_type, metric = self._classify_question(question)
        
        logger.info(f"Classified as: {viz_type}, metric: {metric}")
        
        # Generate appropriate visualization
        if viz_type == 'time_series':
            fig = self._plot_time_series(metric, question)
        elif viz_type == 'channel_comparison':
            fig = self._plot_channel_comparison(metric, question)
        elif viz_type == 'response_curve':
            if curves:
                fig = self._plot_response_curves(curves, question)
            else:
                fig = self._plot_placeholder("Response curves not yet generated")
        elif viz_type == 'optimization':
            if optimization_result is not None:
                fig = self._plot_optimization(optimization_result, question)
            else:
                fig = self._plot_placeholder("Optimization not yet run")
        elif viz_type == 'attribution':
            fig = self._plot_attribution(metric, question)
        elif viz_type == 'correlation':
            fig = self._plot_correlation(question)
        else:
            fig = self._plot_default_overview(question)
        
        # Generate explanation
        explanation = self._generate_explanation(viz_type, metric)
        
        return fig, explanation
    
    def _classify_question(self, question: str) -> Tuple[str, str]:
        """
        Classify question to determine visualization type and metric.
        
        Returns
        -------
        Tuple[str, str]
            (visualization_type, metric_name)
        """
        q_lower = question.lower()
        
        # Extract metric
        metric = None
        for m in self.metrics.get_all_metrics():
            if m in q_lower or self.metrics.DEFINITIONS[m]['name'].lower() in q_lower:
                metric = m
                break
        
        # Determine visualization type
        if any(kw in q_lower for kw in ['over time', 'trend', 'time series', 'weekly', 'monthly']):
            return 'time_series', metric or 'contribution'
        
        elif any(kw in q_lower for kw in ['by channel', 'compare', 'across channels', 'rank']):
            return 'channel_comparison', metric or 'roi'
        
        elif any(kw in q_lower for kw in ['curve', 'saturation', 'response', 'diminishing']):
            return 'response_curve', metric or 'saturation'
        
        elif any(kw in q_lower for kw in ['optimal', 'allocation', 'budget', 'optimize']):
            return 'optimization', metric or 'allocation'
        
        elif any(kw in q_lower for kw in ['attribution', 'decomposition', 'breakdown', 'contribution']):
            return 'attribution', metric or 'contribution'
        
        elif any(kw in q_lower for kw in ['correlation', 'relationship', 'vs']):
            return 'correlation', metric
        
        else:
            return 'overview', metric
    
    def _plot_time_series(self, metric: str, question: str) -> go.Figure:
        """Plot time series for metric"""
        fig = go.Figure()
        
        for channel in self.data['channel'].unique():
            channel_data = self.data[self.data['channel'] == channel].sort_values('week_monday')
            
            # Determine y-axis based on metric
            if metric == 'roi' or 'roi' in question.lower():
                y_values = channel_data['predicted'] / channel_data['spend']
                y_label = 'ROI'
            elif metric == 'roas' or 'roas' in question.lower():
                y_values = channel_data['predicted'] / channel_data['spend']
                y_label = 'ROAS'
            elif 'spend' in question.lower():
                y_values = channel_data['spend']
                y_label = 'Spend ($)'
            else:  # Default to contribution
                y_values = channel_data['predicted']
                y_label = 'Contribution'
            
            fig.add_trace(go.Scatter(
                x=channel_data['week_monday'],
                y=y_values,
                mode='lines+markers',
                name=channel,
                hovertemplate=f'{channel}<br>Week: %{{x}}<br>{y_label}: %{{y:,.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=question,
            xaxis_title="Week",
            yaxis_title=y_label,
            hovermode='x unified',
            template='plotly_white',
            height=600,
            width=1200,
            legend=dict(x=1.02, y=1)
        )
        
        return fig
    
    def _plot_channel_comparison(self, metric: str, question: str) -> go.Figure:
        """Plot channel comparison bar chart"""
        # Aggregate by channel
        channel_agg = self.data.groupby('channel').agg({
            'spend': 'sum',
            'predicted': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        # Calculate metric
        if metric == 'roi' or 'roi' in question.lower():
            channel_agg['metric_value'] = channel_agg['predicted'] / channel_agg['spend']
            metric_label = 'ROI'
            format_str = '.2f'
        elif metric == 'roas' or 'roas' in question.lower():
            channel_agg['metric_value'] = channel_agg['predicted'] / channel_agg['spend']
            metric_label = 'ROAS'
            format_str = '.2f'
        elif metric == 'cpm' or 'cpm' in question.lower():
            channel_agg['metric_value'] = (channel_agg['spend'] / channel_agg['impressions']) * 1000
            metric_label = 'CPM ($)'
            format_str = '.2f'
        elif 'spend' in question.lower():
            channel_agg['metric_value'] = channel_agg['spend']
            metric_label = 'Total Spend ($)'
            format_str = ',.0f'
        else:  # Default to contribution
            channel_agg['metric_value'] = channel_agg['predicted']
            metric_label = 'Total Contribution'
            format_str = ',.0f'
        
        # Sort
        channel_agg = channel_agg.sort_values('metric_value', ascending=False)
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=channel_agg['channel'],
            y=channel_agg['metric_value'],
            marker_color='steelblue',
            text=[f'{v:{format_str}}' for v in channel_agg['metric_value']],
            textposition='outside',
            hovertemplate='%{x}<br>' + metric_label + ': %{y:' + format_str + '}<extra></extra>'
        ))
        
        fig.update_layout(
            title=question,
            xaxis_title="Channel",
            yaxis_title=metric_label,
            template='plotly_white',
            height=600,
            width=1000,
            showlegend=False
        )
        
        return fig
    
    def _plot_response_curves(self, curves: Dict, question: str) -> go.Figure:
        """Plot response curves"""
        # Extract channel if specified in question
        q_lower = question.lower()
        target_channels = [ch for ch in curves.keys() if ch.lower() in q_lower]
        
        if not target_channels:
            target_channels = list(curves.keys())[:6]  # Show top 6
        
        # Create subplots if multiple channels
        if len(target_channels) > 1:
            cols = 2
            rows = (len(target_channels) + 1) // 2
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=target_channels,
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            for idx, channel in enumerate(target_channels):
                row = idx // cols + 1
                col = idx % cols + 1
                
                params = curves[channel]
                spend_range = np.linspace(0, params['saturation'] * 2.5, 100)
                
                def predict(x):
                    return params['bottom'] + (params['top'] - params['bottom']) * \
                           x**params['slope'] / (params['saturation']**params['slope'] + x**params['slope'])
                
                response = [predict(s) for s in spend_range]
                
                fig.add_trace(
                    go.Scatter(x=spend_range, y=response, mode='lines', name=channel, showlegend=False),
                    row=row, col=col
                )
                
                # Saturation point
                fig.add_trace(
                    go.Scatter(
                        x=[params['saturation']],
                        y=[predict(params['saturation'])],
                        mode='markers',
                        marker=dict(color='red', size=8),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                fig.update_xaxes(title_text="Spend", row=row, col=col)
                fig.update_yaxes(title_text="Response", row=row, col=col)
            
            fig.update_layout(
                title_text=question,
                height=400 * rows,
                width=1200,
                template='plotly_white'
            )
        
        else:
            # Single channel detailed curve
            channel = target_channels[0]
            params = curves[channel]
            
            spend_range = np.linspace(0, params['saturation'] * 3, 200)
            
            def predict(x):
                return params['bottom'] + (params['top'] - params['bottom']) * \
                       x**params['slope'] / (params['saturation']**params['slope'] + x**params['slope'])
            
            response = [predict(s) for s in spend_range]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=spend_range,
                y=response,
                mode='lines',
                name='Response Curve',
                line=dict(color='steelblue', width=3)
            ))
            
            # Saturation point
            sat_response = predict(params['saturation'])
            fig.add_trace(go.Scatter(
                x=[params['saturation']],
                y=[sat_response],
                mode='markers',
                name='Saturation Point',
                marker=dict(color='red', size=12, symbol='diamond')
            ))
            
            fig.add_vline(
                x=params['saturation'],
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text=f"Saturation: ${params['saturation']:,.0f}"
            )
            
            fig.update_layout(
                title=f"{channel} - {question}",
                xaxis_title="Spend ($)",
                yaxis_title="Response",
                template='plotly_white',
                height=600,
                width=1000
            )
        
        return fig
    
    def _plot_optimization(self, result_df: pd.DataFrame, question: str) -> go.Figure:
        """Plot optimization results"""
        df = result_df.sort_values('total_spend', ascending=False)
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Budget Allocation', 'ROI by Channel', 'Spend vs Response'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}, {'type': 'scatter'}]],
            horizontal_spacing=0.12
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=df['channel'], values=df['total_spend'], textinfo='label+percent'),
            row=1, col=1
        )
        
        # ROI bar
        fig.add_trace(
            go.Bar(
                x=df['channel'],
                y=df['roi'],
                marker_color='lightblue',
                text=[f'{roi:.2f}x' for roi in df['roi']],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # Scatter
        fig.add_trace(
            go.Scatter(
                x=df['total_spend'],
                y=df['total_response'],
                mode='markers+text',
                text=df['channel'],
                textposition='top center',
                marker=dict(size=12)
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title_text=question,
            height=500,
            width=1600,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _plot_attribution(self, metric: str, question: str) -> go.Figure:
        """Plot attribution/decomposition"""
        channel_agg = self.data.groupby('channel')['predicted'].sum().sort_values(ascending=False)
        
        fig = go.Figure(go.Waterfall(
            name="Contribution",
            orientation="v",
            measure=["relative"] * len(channel_agg) + ["total"],
            x=list(channel_agg.index) + ["Total"],
            y=list(channel_agg.values) + [0],
            text=[f'{v:,.0f}' for v in channel_agg.values] + [f'{channel_agg.sum():,.0f}'],
            textposition="outside"
        ))
        
        fig.update_layout(
            title=question,
            xaxis_title="Channel",
            yaxis_title="Contribution",
            template='plotly_white',
            height=600,
            width=1000
        )
        
        return fig
    
    def _plot_correlation(self, question: str) -> go.Figure:
        """Plot correlation/relationship"""
        # Aggregate by channel
        channel_data = self.data.groupby('channel').agg({
            'spend': 'sum',
            'predicted': 'sum',
            'impressions': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=channel_data['spend'],
            y=channel_data['predicted'],
            mode='markers+text',
            text=channel_data['channel'],
            textposition='top center',
            marker=dict(size=12, color='steelblue'),
            hovertemplate='%{text}<br>Spend: $%{x:,.0f}<br>Response: %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=question,
            xaxis_title="Spend ($)",
            yaxis_title="Response",
            template='plotly_white',
            height=600,
            width=1000
        )
        
        return fig
    
    def _plot_default_overview(self, question: str) -> go.Figure:
        """Default overview visualization"""
        return self._plot_channel_comparison('roi', "ROI by Channel - Overview")
    
    def _plot_placeholder(self, message: str) -> go.Figure:
        """Placeholder when data not available"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        
        fig.update_layout(
            title="Data Not Available",
            template='plotly_white',
            height=400,
            width=800
        )
        
        return fig
    
    def _generate_explanation(self, viz_type: str, metric: Optional[str]) -> str:
        """Generate explanation with metric definitions"""
        explanation = f"**Visualization Type**: {viz_type.replace('_', ' ').title()}\n\n"
        
        if metric:
            metric_def = self.metrics.get_definition(metric)
            explanation += f"**Metric**: {metric_def['name']}\n\n"
            explanation += f"{metric_def['explanation']}\n\n"
            explanation += f"**Formula**: {metric_def['formula']}\n\n"
            
            if metric_def['interpretation']:
                explanation += "**How to Interpret**:\n"
                for level, desc in metric_def['interpretation'].items():
                    explanation += f"- {level.title()}: {desc}\n"
        
        return explanation

