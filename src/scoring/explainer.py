"""Score explanation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class FeatureContribution:
    """Contribution of a single feature to the score."""
    feature_name: str
    value: float
    contribution: float
    direction: str  # "increases" or "decreases"
    percentile: float  # Where this value falls in historical distribution
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature_name,
            "value": self.value,
            "contribution": self.contribution,
            "direction": self.direction,
            "percentile": self.percentile,
        }


class ScoreExplainer:
    """Explain why a score is high or low."""
    
    def __init__(
        self,
        feature_means: Optional[Dict[str, float]] = None,
        feature_stds: Optional[Dict[str, float]] = None,
    ):
        """Initialize explainer.
        
        Args:
            feature_means: Historical mean values per feature
            feature_stds: Historical std values per feature
        """
        self.feature_means = feature_means or {}
        self.feature_stds = feature_stds or {}
    
    def fit(self, X: pd.DataFrame):
        """Compute historical statistics from training data.
        
        Args:
            X: Training feature DataFrame
        """
        self.feature_means = X.mean().to_dict()
        self.feature_stds = X.std().to_dict()
        
        # Store percentile info
        self._percentiles = {}
        for col in X.columns:
            self._percentiles[col] = np.percentile(X[col].dropna(), [10, 25, 50, 75, 90])
    
    def explain(
        self,
        features: pd.Series,
        contributions: pd.Series,
        top_k: int = 5,
    ) -> List[FeatureContribution]:
        """Explain feature contributions for a single prediction.
        
        Args:
            features: Feature values
            contributions: Feature contributions (e.g., from SHAP or tree)
            top_k: Number of top features to return
            
        Returns:
            List of FeatureContribution objects
        """
        # Sort by absolute contribution
        abs_contrib = contributions.abs()
        top_indices = abs_contrib.nlargest(top_k).index
        
        explanations = []
        for idx in top_indices:
            feature_name = idx
            value = features[idx]
            contrib = contributions[idx]
            
            # Determine direction
            direction = "increases" if contrib > 0 else "decreases"
            
            # Compute percentile
            percentile = self._compute_percentile(feature_name, value)
            
            explanations.append(FeatureContribution(
                feature_name=feature_name,
                value=float(value),
                contribution=float(contrib),
                direction=direction,
                percentile=percentile,
            ))
        
        return explanations
    
    def _compute_percentile(self, feature_name: str, value: float) -> float:
        """Compute percentile of a value in historical distribution."""
        if feature_name not in self._percentiles:
            return 50.0
        
        pcts = self._percentiles[feature_name]
        
        if value <= pcts[0]:
            return 5.0
        elif value <= pcts[1]:
            return 25.0 - 15 * (pcts[1] - value) / (pcts[1] - pcts[0])
        elif value <= pcts[2]:
            return 50.0 - 25 * (pcts[2] - value) / (pcts[2] - pcts[1])
        elif value <= pcts[3]:
            return 75.0 - 25 * (pcts[3] - value) / (pcts[3] - pcts[2])
        elif value <= pcts[4]:
            return 90.0 - 15 * (pcts[4] - value) / (pcts[4] - pcts[3])
        else:
            return 95.0
    
    def generate_narrative(
        self,
        explanations: List[FeatureContribution],
        score: float,
    ) -> str:
        """Generate human-readable explanation.
        
        Args:
            explanations: List of feature contributions
            score: Overall anomaly score
            
        Returns:
            Narrative explanation string
        """
        if score < 0.3:
            severity = "low"
        elif score < 0.5:
            severity = "moderate"
        elif score < 0.7:
            severity = "elevated"
        elif score < 0.85:
            severity = "high"
        else:
            severity = "critical"
        
        narrative = f"This event window has a {severity} anomaly score ({score:.2f}). "
        
        if not explanations:
            return narrative + "No significant feature contributions detected."
        
        # Top contributors
        increasing = [e for e in explanations if e.direction == "increases"]
        
        if increasing:
            top = increasing[0]
            narrative += f"The primary driver is {top.feature_name} "
            narrative += f"(value: {top.value:.2f}, {top.percentile:.0f}th percentile), "
            narrative += f"which increases the anomaly score. "
        
        if len(increasing) > 1:
            other_features = [e.feature_name for e in increasing[1:3]]
            narrative += f"Other contributing factors: {', '.join(other_features)}."
        
        return narrative
    
    def to_json(
        self,
        explanations: List[FeatureContribution],
        score: float,
        include_narrative: bool = True,
    ) -> Dict[str, Any]:
        """Convert explanation to JSON-serializable dict.
        
        Args:
            explanations: List of feature contributions
            score: Overall score
            include_narrative: Whether to include narrative
            
        Returns:
            JSON-ready dict
        """
        result = {
            "score": score,
            "features": [e.to_dict() for e in explanations],
        }
        
        if include_narrative:
            result["narrative"] = self.generate_narrative(explanations, score)
        
        return result
