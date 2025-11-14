"""Data validation utilities"""

from typing import List
import pandas as pd


def validate_data(
    data: pd.DataFrame,
    required_columns: List[str] = None
) -> tuple[bool, str]:
    """
    Validate MMM data format.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    required_columns : List[str], optional
        Required column names
        
    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    if required_columns is None:
        required_columns = ['week_monday', 'channel', 'spend', 'impressions', 'predicted']
    
    # Check DataFrame
    if not isinstance(data, pd.DataFrame):
        return False, "Data must be a pandas DataFrame"
    
    # Check empty
    if len(data) == 0:
        return False, "Data is empty"
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check data types
    if 'spend' in data.columns and not pd.api.types.is_numeric_dtype(data['spend']):
        return False, "'spend' column must be numeric"
    
    if 'impressions' in data.columns and not pd.api.types.is_numeric_dtype(data['impressions']):
        return False, "'impressions' column must be numeric"
    
    if 'predicted' in data.columns and not pd.api.types.is_numeric_dtype(data['predicted']):
        return False, "'predicted' column must be numeric"
    
    # Check for nulls in critical columns
    for col in ['spend', 'predicted']:
        if col in data.columns and data[col].isnull().any():
            return False, f"'{col}' column contains null values"
    
    # Check negative values
    for col in ['spend', 'impressions', 'predicted']:
        if col in data.columns and (data[col] < 0).any():
            return False, f"'{col}' column contains negative values"
    
    return True, "Data validation passed"


def validate_constraints(
    constraints: dict,
    channels: List[str]
) -> tuple[bool, str]:
    """
    Validate optimization constraints.
    
    Parameters
    ----------
    constraints : dict
        Constraint dictionary
    channels : List[str]
        Available channels
        
    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    for channel, bounds in constraints.items():
        # Check channel exists
        if channel not in channels:
            return False, f"Unknown channel in constraints: {channel}"
        
        # Check bounds structure
        if not isinstance(bounds, dict):
            return False, f"Constraints for {channel} must be a dictionary"
        
        if 'lower' not in bounds or 'upper' not in bounds:
            return False, f"Constraints for {channel} must have 'lower' and 'upper' keys"
        
        # Check values
        lower = bounds['lower']
        upper = bounds['upper']
        
        if lower < 0:
            return False, f"Lower bound for {channel} cannot be negative"
        
        if upper < lower:
            return False, f"Upper bound for {channel} must be >= lower bound"
    
    return True, "Constraints validation passed"

