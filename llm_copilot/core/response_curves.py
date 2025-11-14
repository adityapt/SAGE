"""
Response Curve Generation using Hill Transformation

Integrates DeepCausalMMM's ResponseCurveFit for production-ready curve generation.
"""

from typing import Dict, List, Optional, Literal
import logging

import pandas as pd
import numpy as np
from deepcausalmmm.postprocess import ResponseCurveFit

logger = logging.getLogger(__name__)


class ResponseCurveGenerator:
    """
    Generate response curves for marketing channels using Hill transformation.
    
    Uses DeepCausalMMM's ResponseCurveFit with week-level data to generate
    saturation curves for budget optimization and strategic planning.
    
    Parameters
    ----------
    data : pd.DataFrame
        Week-level data with columns: week_monday, channel, spend, impressions, predicted
    bottom_param : bool, default=False
        Whether to fit non-zero intercept (typically False for MMM)
    date_col : str, default='week_monday'
        Name of the date column
        
    Attributes
    ----------
    curves : Dict[str, Dict]
        Fitted curves by channel with parameters: top, bottom, saturation, slope, r_2
        
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'week_monday': pd.date_range('2024-01-01', periods=52, freq='W-MON'),
    ...     'channel': ['TV'] * 52,
    ...     'spend': np.random.uniform(10000, 50000, 52),
    ...     'impressions': np.random.uniform(1000000, 5000000, 52),
    ...     'predicted': np.random.uniform(500000, 2000000, 52)
    ... })
    >>> generator = ResponseCurveGenerator(data)
    >>> curves = generator.fit_curves(channels=['TV'])
    >>> print(f"TV Slope: {curves['TV']['slope']:.2f}")
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        bottom_param: bool = False,
        date_col: str = 'week_monday'
    ):
        self.data = data.copy()
        self.bottom_param = bottom_param
        self.date_col = date_col
        self.curves: Dict[str, Dict] = {}
        
        # Validate required columns
        required_cols = [date_col, 'channel', 'spend', 'impressions', 'predicted']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        logger.info(f"Initialized ResponseCurveGenerator with {len(self.data)} rows")
    
    def fit_curves(
        self,
        channels: Optional[List[str]] = None,
        *,
        model_level: Literal['Overall', 'DMA'] = 'Overall',
        generate_figures: bool = False,
        save_figures: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Fit response curves for specified channels.
        
        Parameters
        ----------
        channels : List[str], optional
            List of channels to fit curves for. If None, fits all channels in data.
        model_level : {'Overall', 'DMA'}, default='Overall'
            Aggregation level for curve fitting
        generate_figures : bool, default=False
            Whether to generate visualization figures
        save_figures : bool, default=False
            Whether to save figures to disk
        output_dir : str, optional
            Directory to save figures (required if save_figures=True)
            
        Returns
        -------
        Dict[str, Dict]
            Dictionary mapping channel names to curve parameters:
            {
                'channel_name': {
                    'top': float,
                    'bottom': float,
                    'saturation': float,
                    'slope': float,
                    'r_2': float,
                    'equation': str
                }
            }
        """
        if channels is None:
            channels = self.data['channel'].unique().tolist()
            
        if save_figures and not output_dir:
            raise ValueError("output_dir required when save_figures=True")
            
        logger.info(f"Fitting curves for {len(channels)} channels: {channels}")
        
        for channel in channels:
            try:
                channel_data = self.data[self.data['channel'] == channel].copy()
                
                if len(channel_data) < 10:
                    logger.warning(f"Insufficient data for {channel}: {len(channel_data)} rows. Skipping.")
                    continue
                
                # Initialize ResponseCurveFit from DeepCausalMMM
                fitter = ResponseCurveFit(
                    data=channel_data,
                    bottom_param=self.bottom_param,
                    model_level=model_level,
                    date_col=self.date_col
                )
                
                # Fit the curve
                output_path = f"{output_dir}/{channel}_response_curve.html" if save_figures else None
                
                fitter.fit(
                    title=f"{channel} Response Curve",
                    x_label="Spend",
                    y_label="Predicted Response",
                    generate_figure=generate_figures,
                    save_figure=save_figures,
                    output_path=output_path,
                    print_r_sqr=True
                )
                
                # Store fitted parameters
                if fitter.fit_flag:
                    self.curves[channel] = {
                        'top': fitter.top,
                        'bottom': fitter.bottom,
                        'saturation': fitter.saturation,
                        'slope': fitter.slope,
                        'r_2': fitter.r_2,
                        'equation': fitter.equation if hasattr(fitter, 'equation') else None
                    }
                    
                    logger.info(
                        f"{channel} - Slope: {fitter.slope:.3f}, "
                        f"Saturation: ${fitter.saturation:,.0f}, "
                        f"R²: {fitter.r_2:.3f}"
                    )
                else:
                    logger.warning(f"Curve fitting failed for {channel}")
                    
            except Exception as e:
                logger.error(f"Error fitting curve for {channel}: {e}")
                continue
        
        logger.info(f"Successfully fitted {len(self.curves)} curves")
        return self.curves
    
    def predict(self, channel: str, spend: float) -> float:
        """
        Predict response for a given channel and spend level.
        
        Parameters
        ----------
        channel : str
            Channel name
        spend : float
            Spend amount
            
        Returns
        -------
        float
            Predicted response
        """
        if channel not in self.curves:
            raise ValueError(f"No curve fitted for channel: {channel}")
        
        params = self.curves[channel]
        
        # Hill equation
        response = params['bottom'] + (params['top'] - params['bottom']) * \
                   spend**params['slope'] / (params['saturation']**params['slope'] + spend**params['slope'])
        
        return response
    
    def get_saturation_level(self, channel: str, current_spend: float) -> float:
        """
        Calculate saturation level (0-1) for a channel at given spend.
        
        Parameters
        ----------
        channel : str
            Channel name
        spend : float
            Current spend level
            
        Returns
        -------
        float
            Saturation level between 0 and 1
        """
        if channel not in self.curves:
            raise ValueError(f"No curve fitted for channel: {channel}")
        
        params = self.curves[channel]
        current_response = self.predict(channel, current_spend)
        max_response = params['top']
        
        saturation = current_response / max_response if max_response > 0 else 0
        return saturation
    
    def export_curves(self) -> pd.DataFrame:
        """
        Export fitted curves as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: channel, top, bottom, saturation, slope, r_2
        """
        if not self.curves:
            logger.warning("No curves fitted yet")
            return pd.DataFrame()
        
        curves_list = []
        for channel, params in self.curves.items():
            row = {'channel': channel, **params}
            if 'equation' in row:
                del row['equation']  # Too long for CSV export
            curves_list.append(row)
        
        return pd.DataFrame(curves_list)

