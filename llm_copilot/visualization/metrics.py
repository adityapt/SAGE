"""
Semantic Layer for MMM Metrics

Provides definitions and explanations for all MMM metrics.
"""

from typing import Dict


class MetricDefinitions:
    """
    Semantic layer for MMM metric definitions and explanations.
    """
    
    DEFINITIONS = {
        'roi': {
            'name': 'Return on Investment (ROI)',
            'formula': '(Total Response / Total Spend)',
            'explanation': 'How many dollars of response you get for every dollar spent. '
                          'ROI of 2.0x means every $1 spent returns $2.',
            'interpretation': {
                'excellent': '> 3.0x',
                'good': '1.5x - 3.0x',
                'acceptable': '1.0x - 1.5x',
                'poor': '< 1.0x (losing money)'
            },
            'units': 'multiplier (e.g., 2.5x)'
        },
        
        'roas': {
            'name': 'Return on Ad Spend (ROAS)',
            'formula': '(Revenue / Ad Spend)',
            'explanation': 'Similar to ROI but specifically for advertising spend. '
                          'ROAS of 4.0 means $4 in revenue for every $1 in ad spend.',
            'interpretation': {
                'excellent': '> 4.0',
                'good': '2.0 - 4.0',
                'acceptable': '1.0 - 2.0',
                'poor': '< 1.0'
            },
            'units': 'ratio'
        },
        
        'contribution': {
            'name': 'Marketing Contribution',
            'formula': 'Model-predicted impact of marketing activity',
            'explanation': 'The incremental response (visits, revenue, conversions) '
                          'attributed to a specific channel or campaign.',
            'interpretation': {
                'high': 'Major driver of business outcomes',
                'medium': 'Meaningful contributor',
                'low': 'Marginal impact'
            },
            'units': 'same as response variable (visits, revenue, etc.)'
        },
        
        'saturation': {
            'name': 'Saturation Level',
            'formula': '(Current Response / Maximum Possible Response)',
            'explanation': 'How close a channel is to its maximum effectiveness. '
                          'High saturation means diminishing returns from additional spend.',
            'interpretation': {
                'high': '> 80% - Near maximum, avoid increases',
                'medium': '50-80% - Room for moderate increases',
                'low': '< 50% - Significant opportunity to scale'
            },
            'units': 'percentage (0-100%)'
        },
        
        'cpa': {
            'name': 'Cost Per Acquisition (CPA)',
            'formula': '(Total Spend / Total Conversions)',
            'explanation': 'How much it costs to acquire one customer or conversion.',
            'interpretation': {
                'excellent': 'Well below target CPA',
                'good': 'At or near target CPA',
                'poor': 'Above target CPA'
            },
            'units': 'currency (e.g., $50)'
        },
        
        'cpm': {
            'name': 'Cost Per Thousand Impressions (CPM)',
            'formula': '(Total Spend / Impressions) * 1000',
            'explanation': 'Cost to reach 1,000 people with your ad.',
            'interpretation': {
                'tv': '$20-$50 typical',
                'digital': '$5-$20 typical',
                'social': '$2-$10 typical'
            },
            'units': 'currency per 1,000 impressions'
        },
        
        'adstock': {
            'name': 'Advertising Carryover Effect',
            'formula': 'Decay rate of advertising impact over time',
            'explanation': 'How long advertising continues to have an effect after it runs. '
                          'TV often has longer adstock (2-4 weeks) than digital (1-2 weeks).',
            'interpretation': {
                'long': '> 4 weeks - Brand building',
                'medium': '2-4 weeks - Typical for TV/video',
                'short': '< 2 weeks - Direct response'
            },
            'units': 'weeks or decay rate'
        },
        
        'baseline': {
            'name': 'Baseline Response',
            'formula': 'Expected response with zero marketing',
            'explanation': 'The organic/non-marketing driven response. '
                          'Includes brand equity, word-of-mouth, seasonal patterns.',
            'interpretation': {
                'high': '> 60% of total - Strong brand',
                'medium': '40-60% - Balanced',
                'low': '< 40% - Marketing dependent'
            },
            'units': 'same as response variable'
        },
        
        'elasticity': {
            'name': 'Marketing Elasticity',
            'formula': '% Change in Response / % Change in Spend',
            'explanation': 'How sensitive response is to changes in spend. '
                          'Elasticity of 0.5 means 10% spend increase yields 5% response increase.',
            'interpretation': {
                'high': '> 0.7 - Very responsive',
                'medium': '0.3-0.7 - Moderate responsiveness',
                'low': '< 0.3 - Saturated or weak response'
            },
            'units': 'dimensionless ratio'
        },
        
        'incrementality': {
            'name': 'Incremental Impact',
            'formula': '(Response with Marketing - Response without) / Response without',
            'explanation': 'The true causal lift from marketing versus what would have happened anyway.',
            'interpretation': {
                'high': '> 50% lift - Strong incremental impact',
                'medium': '20-50% lift - Moderate impact',
                'low': '< 20% lift - Weak incrementality'
            },
            'units': 'percentage lift'
        },
        
        'share_of_voice': {
            'name': 'Share of Voice (SOV)',
            'formula': '(Your Spend / Total Category Spend)',
            'explanation': 'Your advertising spend as a percentage of total category advertising.',
            'interpretation': {
                'leader': '> 30% - Market leader',
                'challenger': '15-30% - Strong presence',
                'follower': '< 15% - Limited visibility'
            },
            'units': 'percentage'
        }
    }
    
    @classmethod
    def get_definition(cls, metric: str) -> Dict:
        """
        Get definition for a metric.
        
        Parameters
        ----------
        metric : str
            Metric name (case-insensitive)
            
        Returns
        -------
        Dict
            Metric definition with name, formula, explanation, interpretation
        """
        metric_lower = metric.lower().replace(' ', '_').replace('-', '_')
        
        # Handle aliases
        aliases = {
            'return_on_investment': 'roi',
            'return_on_ad_spend': 'roas',
            'cost_per_acquisition': 'cpa',
            'cost_per_action': 'cpa',
            'cost_per_mille': 'cpm',
            'carryover': 'adstock',
            'decay': 'adstock'
        }
        
        metric_key = aliases.get(metric_lower, metric_lower)
        
        if metric_key in cls.DEFINITIONS:
            return cls.DEFINITIONS[metric_key]
        else:
            return {
                'name': metric,
                'formula': 'Unknown',
                'explanation': f'Metric "{metric}" not found in definitions.',
                'interpretation': {},
                'units': 'unknown'
            }
    
    @classmethod
    def explain_metric(cls, metric: str, value: float = None) -> str:
        """
        Generate natural language explanation of a metric.
        
        Parameters
        ----------
        metric : str
            Metric name
        value : float, optional
            Metric value for contextualized explanation
            
        Returns
        -------
        str
            Natural language explanation
        """
        definition = cls.get_definition(metric)
        
        explanation = f"**{definition['name']}**\n\n"
        explanation += f"{definition['explanation']}\n\n"
        explanation += f"**Formula**: {definition['formula']}\n"
        
        if value is not None:
            explanation += f"\n**Your Value**: {value:.2f} {definition['units']}\n"
            
            # Add interpretation based on value
            if 'interpretation' in definition and definition['interpretation']:
                explanation += "\n**Interpretation**:\n"
                for level, desc in definition['interpretation'].items():
                    explanation += f"- {level.title()}: {desc}\n"
        
        return explanation
    
    @classmethod
    def get_all_metrics(cls) -> list:
        """Get list of all available metrics"""
        return list(cls.DEFINITIONS.keys())
    
    @classmethod
    def search_metrics(cls, query: str) -> list:
        """
        Search for metrics by keyword.
        
        Parameters
        ----------
        query : str
            Search term
            
        Returns
        -------
        list
            Matching metric keys
        """
        query_lower = query.lower()
        matches = []
        
        for key, definition in cls.DEFINITIONS.items():
            if (query_lower in key.lower() or
                query_lower in definition['name'].lower() or
                query_lower in definition['explanation'].lower()):
                matches.append(key)
        
        return matches

