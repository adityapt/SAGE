"""
Budget Optimization using SLSQP with Response Curves

Production-ready optimizer using Hill transformation response curves
for marketing budget allocation across channels.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
import scipy.optimize as op

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """
    Result from budget optimization.
    
    Attributes
    ----------
    success : bool
        Whether optimization converged successfully
    allocation : Dict[str, float]
        Optimal spend allocation by channel
    predicted_response : float
        Total predicted response at optimal allocation
    by_channel : pd.DataFrame
        Detailed results by channel
    by_kpi : pd.DataFrame
        Results by business KPI (if multiple KPIs)
    message : str
        Optimization status message
    """
    success: bool
    allocation: Dict[str, float]
    predicted_response: float
    by_channel: pd.DataFrame
    by_kpi: Optional[pd.DataFrame] = None
    message: str = ""


class BudgetOptimizer:
    """
    Optimize marketing budget allocation using response curves.
    
    Uses SLSQP optimization with Hill transformation curves to find
    optimal spend allocation that maximizes total response subject
    to business constraints.
    
    Parameters
    ----------
    budget : float
        Total budget to allocate
    channels : List[str]
        List of channel names to include in optimization
    response_curves : Dict[str, Dict]
        Response curve parameters by channel from ResponseCurveGenerator
    num_weeks : int, default=52
        Number of weeks for planning horizon
    business_kpis : List[str], optional
        Business KPIs to optimize (e.g., ['revenue', 'visits'])
        
    Examples
    --------
    >>> optimizer = BudgetOptimizer(
    ...     budget=1000000,
    ...     channels=['TV', 'Search', 'Social'],
    ...     response_curves=curves,
    ...     num_weeks=52
    ... )
    >>> optimizer.set_constraints({
    ...     'TV': {'lower': 50000, 'upper': 500000},
    ...     'Search': {'lower': 100000, 'upper': 400000}
    ... })
    >>> result = optimizer.optimize()
    >>> print(result.allocation)
    """
    
    def __init__(
        self,
        budget: float,
        channels: List[str],
        response_curves: Dict[str, Dict],
        *,
        num_weeks: int = 52,
        business_kpis: Optional[List[str]] = None,
        method: str = 'trust-constr'
    ):
        self.budget = budget
        self.channels = channels
        self.response_curves = response_curves
        self.num_weeks = num_weeks
        self.business_kpis = business_kpis or []
        self.method = method
        
        # Validate method
        valid_methods = ['trust-constr', 'SLSQP', 'differential_evolution', 'hybrid']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
        
        # Validate channels have curves
        missing_curves = [ch for ch in channels if ch not in response_curves]
        if missing_curves:
            raise ValueError(f"Missing response curves for channels: {missing_curves}")
        
        # Initialize constraints
        self.constraints_df: Optional[pd.DataFrame] = None
        
        logger.info(
            f"Initialized BudgetOptimizer: "
            f"Budget=${budget:,.0f}, Channels={len(channels)}, Weeks={num_weeks}, Method={method}"
        )
    
    def set_constraints(self, constraints: Dict[str, Dict[str, float]]) -> None:
        """
        Set spend constraints for channels.
        
        Parameters
        ----------
        constraints : Dict[str, Dict[str, float]]
            Channel constraints: {'channel': {'lower': min_spend, 'upper': max_spend}}
        """
        constraints_list = []
        for channel in self.channels:
            if channel in constraints:
                constraints_list.append({
                    'channel': channel,
                    'lower': constraints[channel].get('lower', 0),
                    'upper': constraints[channel].get('upper', self.budget)
                })
            else:
                # Default: no minimum, max is total budget
                constraints_list.append({
                    'channel': channel,
                    'lower': 0,
                    'upper': self.budget
                })
        
        self.constraints_df = pd.DataFrame(constraints_list)
        
        # Validate constraints
        invalid = self.constraints_df[self.constraints_df['upper'] == 0]
        if len(invalid) > 0:
            raise ValueError(f"Upper constraints cannot be 0 for channels: {invalid['channel'].tolist()}")
        
        # Ensure upper doesn't exceed budget
        self.constraints_df['upper'] = np.minimum(self.constraints_df['upper'], self.budget)
        
        # Ensure lower doesn't exceed upper
        self.constraints_df['lower'] = np.where(
            self.constraints_df['lower'] > self.constraints_df['upper'],
            0,
            self.constraints_df['lower']
        )
        
        logger.info(f"Set constraints for {len(self.channels)} channels")
    
    def _predict_response(self, spend: float, channel: str) -> float:
        """
        Predict response using Hill equation.
        
        Parameters
        ----------
        spend : float
            Weekly spend amount
        channel : str
            Channel name
            
        Returns
        -------
        float
            Predicted response
        """
        params = self.response_curves[channel]
        bottom = params['bottom']
        top = params['top']
        saturation = params['saturation']
        slope = params['slope']
        
        # Handle overflow with safe exponentiation
        try:
            slp_prm = spend ** slope
        except OverflowError:
            slp_prm = np.where(
                np.power(spend, slope, dtype=np.float64) >= spend * 10,
                spend * 10,
                np.power(spend, slope, dtype=np.float64)
            )
        
        try:
            sat_slp = saturation ** slope
        except OverflowError:
            sat_slp = np.where(
                np.power(saturation, slope, dtype=np.float64) >= saturation * 10,
                saturation * 10,
                np.power(saturation, slope, dtype=np.float64)
            )
        
        response = bottom + (top - bottom) * slp_prm / (sat_slp + slp_prm)
        return response
    
    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function to minimize (negative of total response).
        
        Parameters
        ----------
        x : np.ndarray
            Array of spend allocations (one per channel)
            
        Returns
        -------
        float
            Negative total response (for minimization)
        """
        # x contains total spend per channel, convert to weekly
        weekly_spend = x / self.num_weeks
        
        # Calculate total response across all channels
        total_response = 0
        for i, channel in enumerate(self.channels):
            response = self._predict_response(weekly_spend[i], channel)
            total_response += response
        
        # Return negative (we minimize, but want to maximize response)
        return -total_response * self.num_weeks
    
    def _constraint_budget(self, x: np.ndarray) -> float:
        """
        Equality constraint: sum of spend equals budget.
        
        Parameters
        ----------
        x : np.ndarray
            Array of spend allocations
            
        Returns
        -------
        float
            Difference from budget (should be 0)
        """
        return np.sum(x) - self.budget
    
    def _get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for optimization.
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each channel
        """
        if self.constraints_df is None:
            # Default: no constraints
            return [(0, self.budget) for _ in self.channels]
        
        bounds = []
        for channel in self.channels:
            row = self.constraints_df[self.constraints_df['channel'] == channel]
            if len(row) > 0:
                bounds.append((row['lower'].iloc[0], row['upper'].iloc[0]))
            else:
                bounds.append((0, self.budget))
        
        return bounds
    
    def _get_initial_guess(self) -> np.ndarray:
        """
        Get initial guess for optimization.
        
        Uses equal allocation as starting point.
        
        Returns
        -------
        np.ndarray
            Initial spend allocation
        """
        # Start with equal allocation
        x0 = np.full(len(self.channels), self.budget / len(self.channels))
        
        # Adjust to respect bounds
        bounds = self._get_bounds()
        for i, (lower, upper) in enumerate(bounds):
            x0[i] = np.clip(x0[i], lower, upper)
        
        # Normalize to match budget
        x0 = x0 * (self.budget / np.sum(x0))
        
        return x0
    
    def optimize(self) -> OptimizationResult:
        """
        Run optimization to find optimal budget allocation.
        
        Returns
        -------
        OptimizationResult
            Optimization results including allocation and predicted response
        """
        logger.info(f"Starting budget optimization with method={self.method}...")
        
        # Get initial guess and bounds
        x0 = self._get_initial_guess()
        bounds = self._get_bounds()
        constraints = {'type': 'eq', 'fun': self._constraint_budget}
        
        # Run optimization based on method
        if self.method == 'differential_evolution':
            result = self._optimize_global(bounds, constraints)
        elif self.method == 'hybrid':
            result = self._optimize_hybrid(bounds, constraints, x0)
        else:
            result = self._optimize_gradient(x0, bounds, constraints, self.method)
        
        if result.success:
            logger.info(f"Optimization converged: {result.message}")
            
            # Extract results
            allocation = {ch: spend for ch, spend in zip(self.channels, result.x)}
            predicted_response = -result.fun  # Negate back to positive
            
            # Build detailed results
            by_channel = self._calculate_by_channel(result.x)
            
            return OptimizationResult(
                success=True,
                allocation=allocation,
                predicted_response=predicted_response,
                by_channel=by_channel,
                message=result.message
            )
        else:
            logger.error(f"Optimization failed: {result.message}")
            
            return OptimizationResult(
                success=False,
                allocation={ch: 0 for ch in self.channels},
                predicted_response=0,
                by_channel=pd.DataFrame(),
                message=result.message
            )
    
    def _optimize_gradient(
        self,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Dict,
        method: str
    ):
        """Gradient-based optimization (trust-constr or SLSQP)"""
        if method == 'trust-constr':
            # Better than SLSQP: more robust convergence
            return op.minimize(
                fun=self._objective,
                x0=x0,
                method='trust-constr',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 500,
                    'verbose': 1,
                    'gtol': 1e-8,
                    'xtol': 1e-10
                }
            )
        else:  # SLSQP
            return op.minimize(
                fun=self._objective,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 400,
                    'disp': True,
                    'ftol': 1e-9
                },
                jac='3-point'
            )
    
    def _optimize_global(self, bounds: List[Tuple[float, float]], constraints: Dict):
        """Global optimization using differential evolution"""
        from scipy.optimize import differential_evolution
        
        logger.info("Running global optimization (may take longer)...")
        return differential_evolution(
            func=self._objective,
            bounds=bounds,
            constraints=constraints,
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=0.01,
            atol=0,
            seed=42,
            polish=True,  # Local refinement at end
            workers=1
        )
    
    def _optimize_hybrid(
        self,
        bounds: List[Tuple[float, float]],
        constraints: Dict,
        x0: np.ndarray
    ):
        """Hybrid: global search + local refinement"""
        from scipy.optimize import differential_evolution
        
        logger.info("Running hybrid optimization (global + local)...")
        
        # Stage 1: Quick global search
        global_result = differential_evolution(
            func=self._objective,
            bounds=bounds,
            constraints=constraints,
            maxiter=100,  # Quick scan
            popsize=10,
            polish=False,
            workers=1
        )
        
        # Stage 2: Local refinement from global optimum
        logger.info("Refining with local optimizer...")
        local_result = op.minimize(
            fun=self._objective,
            x0=global_result.x,
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'verbose': 1}
        )
        
        return local_result
    
    def _calculate_by_channel(self, optimal_spend: np.ndarray) -> pd.DataFrame:
        """
        Calculate detailed metrics by channel.
        
        Parameters
        ----------
        optimal_spend : np.ndarray
            Optimal spend allocation
            
        Returns
        -------
        pd.DataFrame
            Results by channel with spend and predicted response
        """
        results = []
        
        weekly_spend = optimal_spend / self.num_weeks
        
        for i, channel in enumerate(self.channels):
            response = self._predict_response(weekly_spend[i], channel)
            total_response = response * self.num_weeks
            
            params = self.response_curves[channel]
            
            results.append({
                'channel': channel,
                'total_spend': optimal_spend[i],
                'weekly_spend': weekly_spend[i],
                'weekly_response': response,
                'total_response': total_response,
                'roi': total_response / optimal_spend[i] if optimal_spend[i] > 0 else 0,
                'saturation': params['saturation'],
                'slope': params['slope']
            })
        
        return pd.DataFrame(results)

