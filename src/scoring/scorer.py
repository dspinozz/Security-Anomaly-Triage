"""Unified scorer combining all scoring methods."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

from ..models.baseline import BaselineScorer, RuleConfig
from ..models.lightgbm_model import LightGBMScorer
from ..models.isolation import IsolationForestScorer


@dataclass
class ScoringResult:
    """Result of scoring a single window or batch."""
    anomaly_score: float
    predicted_class: Optional[str] = None
    severity: str = "low"
    top_features: List[Dict] = field(default_factory=list)
    rule_triggers: List[str] = field(default_factory=list)
    model_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_score": self.anomaly_score,
            "predicted_class": self.predicted_class,
            "severity": self.severity,
            "top_features": self.top_features,
            "rule_triggers": self.rule_triggers,
            "model_scores": self.model_scores,
        }


class UnifiedScorer:
    """Combines baseline rules, LightGBM, and Isolation Forest scoring."""
    
    def __init__(
        self,
        baseline_scorer: Optional[BaselineScorer] = None,
        lgb_scorer: Optional[LightGBMScorer] = None,
        if_scorer: Optional[IsolationForestScorer] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize unified scorer.
        
        Args:
            baseline_scorer: Rule-based scorer
            lgb_scorer: LightGBM supervised scorer (optional)
            if_scorer: Isolation Forest unsupervised scorer
            weights: Weights for combining scores
        """
        self.baseline_scorer = baseline_scorer or BaselineScorer()
        self.lgb_scorer = lgb_scorer
        self.if_scorer = if_scorer
        
        # Default weights (sum to 1)
        self.weights = weights or {
            "baseline": 0.2,
            "lgb": 0.5,
            "isolation": 0.3,
        }
        
        # Adjust weights if some models are missing
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Adjust weights based on available models."""
        available_weight = 0.0
        
        if self.baseline_scorer:
            available_weight += self.weights.get("baseline", 0)
        if self.lgb_scorer:
            available_weight += self.weights.get("lgb", 0)
        if self.if_scorer:
            available_weight += self.weights.get("isolation", 0)
        
        if available_weight > 0:
            for key in self.weights:
                self.weights[key] /= available_weight
    
    def score(
        self,
        features: pd.DataFrame,
        return_explanations: bool = True,
        top_k_features: int = 5,
    ) -> pd.DataFrame:
        """Score feature windows.
        
        Args:
            features: DataFrame with feature columns
            return_explanations: Whether to include feature explanations
            top_k_features: Number of top features to return
            
        Returns:
            DataFrame with scoring results
        """
        results = features.copy()
        
        # Baseline scoring
        baseline_result = self.baseline_scorer.score(features)
        results["baseline_score"] = baseline_result["baseline_score"]
        results["triggered_rules"] = baseline_result["triggered_rules"]
        results["rule_severity"] = baseline_result["severity"]
        
        # LightGBM scoring (if available)
        if self.lgb_scorer is not None:
            if return_explanations:
                lgb_result = self.lgb_scorer.score_with_explanation(
                    features, top_k=top_k_features
                )
                results["lgb_score"] = lgb_result["anomaly_score"]
                results["lgb_top_features"] = lgb_result["top_features"]
            else:
                results["lgb_score"] = self.lgb_scorer.predict_proba(features)
        else:
            results["lgb_score"] = 0.0
        
        # Isolation Forest scoring (if available)
        if self.if_scorer is not None:
            if_result = self.if_scorer.score(features)
            results["if_score"] = if_result["combined_score"]
        else:
            results["if_score"] = 0.0
        
        # Compute combined score
        results["anomaly_score"] = (
            self.weights.get("baseline", 0) * results["baseline_score"] +
            self.weights.get("lgb", 0) * results["lgb_score"] +
            self.weights.get("isolation", 0) * results["if_score"]
        ).clip(0, 1)
        
        # Determine severity based on combined score
        results["severity"] = results["anomaly_score"].apply(self._score_to_severity)
        
        return results
    
    def _score_to_severity(self, score: float) -> str:
        """Convert score to severity level."""
        if score >= 0.85:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        elif score >= 0.3:
            return "low"
        else:
            return "info"
    
    def score_single(
        self,
        features: pd.Series,
        return_explanation: bool = True,
    ) -> ScoringResult:
        """Score a single window.
        
        Args:
            features: Series with feature values
            return_explanation: Whether to include explanation
            
        Returns:
            ScoringResult object
        """
        # Convert to DataFrame for consistent processing
        df = pd.DataFrame([features])
        
        result = self.score(df, return_explanations=return_explanation)
        row = result.iloc[0]
        
        return ScoringResult(
            anomaly_score=float(row["anomaly_score"]),
            severity=row["severity"],
            rule_triggers=row.get("triggered_rules", []),
            top_features=row.get("lgb_top_features", []) if return_explanation else [],
            model_scores={
                "baseline": float(row["baseline_score"]),
                "lgb": float(row["lgb_score"]),
                "isolation": float(row["if_score"]),
            },
        )
    
    def get_top_alerts(
        self,
        features: pd.DataFrame,
        limit: int = 50,
        min_score: float = 0.5,
    ) -> pd.DataFrame:
        """Get top alerts sorted by score.
        
        Args:
            features: DataFrame with features
            limit: Maximum number of alerts to return
            min_score: Minimum score threshold
            
        Returns:
            DataFrame with top alerts
        """
        scored = self.score(features, return_explanations=True)
        
        # Filter by threshold
        alerts = scored[scored["anomaly_score"] >= min_score]
        
        # Sort by score descending
        alerts = alerts.sort_values("anomaly_score", ascending=False)
        
        # Limit
        return alerts.head(limit)
    
    def compute_alert_reduction(
        self,
        features: pd.DataFrame,
        baseline_threshold: float = 0.0,
        model_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Compute alert reduction metrics.
        
        Args:
            features: DataFrame with features
            baseline_threshold: Threshold for baseline alerts
            model_threshold: Threshold for model alerts
            
        Returns:
            Dict with reduction metrics
        """
        scored = self.score(features, return_explanations=False)
        
        # Baseline: anything above baseline threshold
        baseline_alerts = (scored["baseline_score"] > baseline_threshold).sum()
        
        # Model: anything above model threshold
        model_alerts = (scored["anomaly_score"] >= model_threshold).sum()
        
        total_events = len(features)
        
        reduction = 1.0 - (model_alerts / max(total_events, 1))
        
        return {
            "total_events": total_events,
            "baseline_alerts": int(baseline_alerts),
            "model_alerts": int(model_alerts),
            "alert_reduction_rate": reduction,
            "alerts_per_1000": (model_alerts / max(total_events, 1)) * 1000,
        }
    
    def save(self, path: Path):
        """Save all models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.lgb_scorer is not None:
            self.lgb_scorer.save(path / "lgb")
        
        if self.if_scorer is not None:
            self.if_scorer.save(path / "isolation")
        
        # Save weights
        import json
        with open(path / "weights.json", "w") as f:
            json.dump(self.weights, f)
    
    @classmethod
    def load(
        cls,
        path: Path,
        rule_config_path: Optional[Path] = None,
    ) -> "UnifiedScorer":
        """Load models from disk."""
        path = Path(path)
        
        import json
        with open(path / "weights.json") as f:
            weights = json.load(f)
        
        # Load baseline scorer
        baseline = None
        if rule_config_path and rule_config_path.exists():
            rule_config = RuleConfig.from_yaml(rule_config_path)
            baseline = BaselineScorer(rule_config)
        else:
            baseline = BaselineScorer()
        
        # Load LightGBM if exists
        lgb_scorer = None
        if (path / "lgb").exists():
            lgb_scorer = LightGBMScorer.load(path / "lgb")
        
        # Load Isolation Forest if exists
        if_scorer = None
        if (path / "isolation").exists():
            if_scorer = IsolationForestScorer.load(path / "isolation")
        
        return cls(
            baseline_scorer=baseline,
            lgb_scorer=lgb_scorer,
            if_scorer=if_scorer,
            weights=weights,
        )
