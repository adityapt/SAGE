"""
DeepCausalMMM Connector

Load data and results from trained DeepCausalMMM models.
"""

from typing import Optional, Dict
from pathlib import Path
import logging
import pickle

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DeepCausalMMMConnector:
    """
    Connector for DeepCausalMMM model outputs.
    
    Loads:
    - Model predictions
    - Channel coefficients
    - Attribution results
    - Response curve parameters
    
    Parameters
    ----------
    model_path : Path
        Path to saved DeepCausalMMM model
        
    Examples
    --------
    >>> connector = DeepCausalMMMConnector("models/mmm_model.pkl")
    >>> data = connector.load_predictions()
    >>> curves = connector.extract_response_curves()
    """
    
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self.results = None
        
        logger.info(f"Initialized DeepCausalMMMConnector: {model_path}")
    
    def load_model(self) -> None:
        """Load trained DeepCausalMMM model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info("Loaded DeepCausalMMM model")
    
    def load_predictions(self) -> pd.DataFrame:
        """
        Load model predictions as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Predictions with columns: date, channel, spend, impressions, predicted
        """
        if self.model is None:
            self.load_model()
        
        # Extract predictions from model
        # This assumes DeepCausalMMM stores predictions in a specific format
        # Adjust based on actual DeepCausalMMM structure
        
        try:
            predictions = self.model.predictions
            df = pd.DataFrame(predictions)
            logger.info(f"Loaded {len(df)} prediction rows")
            return df
        except AttributeError:
            logger.warning("Model does not have predictions attribute. Using alternative method.")
            return self._extract_predictions_alternative()
    
    def extract_response_curves(self) -> Dict[str, Dict]:
        """
        Extract response curve parameters from model.
        
        Returns
        -------
        Dict[str, Dict]
            Response curves by channel with Hill parameters
        """
        if self.model is None:
            self.load_model()
        
        curves = {}
        
        # Extract Hill parameters from DeepCausalMMM
        # Adjust based on actual model structure
        try:
            for channel in self.model.channels:
                params = self.model.get_channel_params(channel)
                curves[channel] = {
                    'slope': params.get('slope', 1.0),
                    'saturation': params.get('saturation', 100000),
                    'bottom': params.get('bottom', 0.0),
                    'top': params.get('top', 10000),
                    'r_2': params.get('r_2', 0.9)
                }
            
            logger.info(f"Extracted response curves for {len(curves)} channels")
            return curves
            
        except AttributeError:
            logger.warning("Could not extract curves from model structure")
            return self._fit_curves_from_predictions()
    
    def extract_channel_contributions(self) -> pd.DataFrame:
        """
        Extract channel contributions over time.
        
        Returns
        -------
        pd.DataFrame
            Channel contributions by date
        """
        if self.model is None:
            self.load_model()
        
        try:
            contributions = self.model.contributions
            df = pd.DataFrame(contributions)
            logger.info("Extracted channel contributions")
            return df
        except AttributeError:
            logger.warning("Model does not have contributions")
            return pd.DataFrame()
    
    def extract_roi_metrics(self) -> Dict[str, float]:
        """
        Extract ROI metrics by channel.
        
        Returns
        -------
        Dict[str, float]
            ROI by channel
        """
        if self.model is None:
            self.load_model()
        
        try:
            roi = {}
            for channel in self.model.channels:
                roi[channel] = self.model.get_channel_roi(channel)
            
            logger.info(f"Extracted ROI for {len(roi)} channels")
            return roi
            
        except AttributeError:
            logger.warning("Could not extract ROI from model")
            return {}
    
    def _extract_predictions_alternative(self) -> pd.DataFrame:
        """Alternative method to extract predictions if direct access fails"""
        # Fallback: construct from available model data
        logger.info("Using alternative prediction extraction")
        
        # This would need to be customized based on actual model structure
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def _fit_curves_from_predictions(self) -> Dict[str, Dict]:
        """Fit response curves from prediction data if not stored in model"""
        from llm_copilot.core.response_curves import ResponseCurveGenerator
        
        logger.info("Fitting curves from predictions")
        
        # Load predictions
        df = self.load_predictions()
        
        if df.empty:
            return {}
        
        # Fit curves
        curve_gen = ResponseCurveGenerator(df)
        curves = curve_gen.fit_all_curves()
        
        return curves
    
    @classmethod
    def from_output_directory(cls, output_dir: Path) -> 'DeepCausalMMMConnector':
        """
        Load from DeepCausalMMM output directory.
        
        Parameters
        ----------
        output_dir : Path
            DeepCausalMMM output directory
            
        Returns
        -------
        DeepCausalMMMConnector
            Connector instance
        """
        output_dir = Path(output_dir)
        
        # Look for model file
        model_files = list(output_dir.glob("*.pkl")) + list(output_dir.glob("*.pt"))
        
        if not model_files:
            raise FileNotFoundError(f"No model file found in {output_dir}")
        
        model_path = model_files[0]
        logger.info(f"Found model: {model_path}")
        
        return cls(model_path)

