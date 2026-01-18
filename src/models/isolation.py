"""Isolation Forest for unsupervised anomaly detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json
import pickle

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class IsolationForestScorer:
    """Unsupervised anomaly scorer using Isolation Forest and LOF."""
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        use_lof: bool = True,
        lof_neighbors: int = 20,
    ):
        """Initialize unsupervised scorer.
        
        Args:
            contamination: Expected fraction of anomalies in training data
            n_estimators: Number of trees in Isolation Forest
            use_lof: Whether to also use Local Outlier Factor
            lof_neighbors: Number of neighbors for LOF
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.use_lof = use_lof
        self.lof_neighbors = lof_neighbors
        
        self.isolation_forest: Optional[IsolationForest] = None
        self.lof: Optional[LocalOutlierFactor] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> "IsolationForestScorer":
        """Fit unsupervised models.
        
        Args:
            X: Feature DataFrame (no labels needed)
            
        Returns:
            Self for chaining
        """
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_scaled)
        
        # Fit LOF (for novelty detection mode, need to refit on new data)
        if self.use_lof:
            self.lof = LocalOutlierFactor(
                n_neighbors=self.lof_neighbors,
                contamination=self.contamination,
                novelty=True,  # Enable prediction on new data
            )
            self.lof.fit(X_scaled)
        
        self._is_fitted = True
        return self
    
    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get anomaly scores.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with anomaly scores
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest scores
        # decision_function returns negative for anomalies
        # We transform to [0, 1] where 1 = most anomalous
        if_raw = self.isolation_forest.decision_function(X_scaled)
        if_scores = self._normalize_scores(-if_raw)  # Negate so anomalies are positive
        
        result = pd.DataFrame(index=X.index)
        result["if_score"] = if_scores
        result["if_anomaly"] = (if_scores > 0.5).astype(int)
        
        # LOF scores
        if self.use_lof and self.lof is not None:
            lof_raw = self.lof.decision_function(X_scaled)
            lof_scores = self._normalize_scores(-lof_raw)
            
            result["lof_score"] = lof_scores
            result["lof_anomaly"] = (lof_scores > 0.5).astype(int)
            
            # Combined score (average of both)
            result["combined_score"] = (result["if_score"] + result["lof_score"]) / 2
        else:
            result["combined_score"] = result["if_score"]
        
        # Overall anomaly determination
        result["is_anomaly"] = (result["combined_score"] > 0.5).astype(int)
        
        return result
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range.
        
        Uses sigmoid-like transformation centered on median.
        """
        # Use percentile-based normalization
        p5, p50, p95 = np.percentile(scores, [5, 50, 95])
        
        # Shift to center on median
        centered = scores - p50
        
        # Scale using IQR-like range
        scale = max(p95 - p50, p50 - p5, 0.1)
        scaled = centered / scale
        
        # Apply sigmoid
        normalized = 1 / (1 + np.exp(-scaled))
        
        return normalized
    
    def score_with_isolation_depth(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score with per-feature isolation depths for explainability.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with scores and feature isolation info
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get per-tree isolation depths
        # This gives insight into which samples are easier to isolate
        depths = np.zeros((len(X), self.n_estimators))
        
        for i, tree in enumerate(self.isolation_forest.estimators_):
            # Get leaf indices for each sample
            leaves = tree.apply(X_scaled)
            
            # Approximate depth from leaf index
            # (leaf nodes at greater depth have higher indices)
            depths[:, i] = np.log2(leaves + 1)
        
        result = self.score(X)
        result["avg_isolation_depth"] = depths.mean(axis=1)
        result["isolation_depth_std"] = depths.std(axis=1)
        
        # Low depth = easy to isolate = anomalous
        result["depth_anomaly_indicator"] = (
            result["avg_isolation_depth"] < np.percentile(result["avg_isolation_depth"], 10)
        ).astype(int)
        
        return result
    
    def get_top_anomalies(
        self, 
        X: pd.DataFrame,
        top_k: int = 50,
    ) -> pd.DataFrame:
        """Get top-k most anomalous samples.
        
        Args:
            X: Feature DataFrame
            top_k: Number of top anomalies to return
            
        Returns:
            DataFrame with top anomalies and their scores
        """
        scores = self.score(X)
        
        # Sort by combined score descending
        top_indices = scores["combined_score"].nlargest(top_k).index
        
        # Get original features and scores
        result = X.loc[top_indices].copy()
        result["anomaly_score"] = scores.loc[top_indices, "combined_score"]
        result["if_score"] = scores.loc[top_indices, "if_score"]
        
        if "lof_score" in scores.columns:
            result["lof_score"] = scores.loc[top_indices, "lof_score"]
        
        return result.sort_values("anomaly_score", ascending=False)
    
    def save(self, path: Path):
        """Save model to disk."""
        if not self._is_fitted:
            raise ValueError("No model to save. Call fit() first.")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save models with pickle
        with open(path / "isolation_forest.pkl", "wb") as f:
            pickle.dump(self.isolation_forest, f)
        
        if self.lof is not None:
            with open(path / "lof.pkl", "wb") as f:
                pickle.dump(self.lof, f)
        
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "use_lof": self.use_lof,
            "lof_neighbors": self.lof_neighbors,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "IsolationForestScorer":
        """Load model from disk."""
        path = Path(path)
        
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        
        scorer = cls(
            contamination=metadata["contamination"],
            n_estimators=metadata["n_estimators"],
            use_lof=metadata["use_lof"],
            lof_neighbors=metadata["lof_neighbors"],
        )
        scorer.feature_names = metadata["feature_names"]
        
        with open(path / "isolation_forest.pkl", "rb") as f:
            scorer.isolation_forest = pickle.load(f)
        
        if scorer.use_lof and (path / "lof.pkl").exists():
            with open(path / "lof.pkl", "rb") as f:
                scorer.lof = pickle.load(f)
        
        with open(path / "scaler.pkl", "rb") as f:
            scorer.scaler = pickle.load(f)
        
        scorer._is_fitted = True
        return scorer
