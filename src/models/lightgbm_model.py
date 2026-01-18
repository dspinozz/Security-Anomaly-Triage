"""LightGBM-based scorer for security anomaly detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score,
    classification_report,
)
from sklearn.calibration import CalibratedClassifierCV


class LightGBMScorer:
    """LightGBM classifier for anomaly scoring with explainability."""
    
    def __init__(
        self,
        params: Optional[Dict] = None,
        calibrate: bool = True,
    ):
        """Initialize LightGBM scorer.
        
        Args:
            params: LightGBM parameters. If None, uses tuned defaults.
            calibrate: Whether to apply Platt/isotonic calibration.
        """
        if lgb is None:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        
        self.params = params or self._default_params()
        self.calibrate = calibrate
        self.model: Optional[lgb.Booster] = None
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
    
    def _default_params(self) -> Dict:
        """Default LightGBM parameters tuned for security data."""
        return {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_child_samples": 20,
            "class_weight": "balanced",  # Handle class imbalance
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
    ) -> "LightGBMScorer":
        """Train the LightGBM model.
        
        Args:
            X: Feature DataFrame
            y: Binary labels (0=normal, 1=attack)
            eval_set: Optional validation set (X_val, y_val)
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Self for chaining
        """
        self.feature_names = list(X.columns)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y)
        
        valid_sets = [train_data]
        if eval_set is not None:
            valid_data = lgb.Dataset(eval_set[0], label=eval_set[1], reference=train_data)
            valid_sets.append(valid_data)
        
        # Extract callback params
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ]
        
        # Train model
        params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.params.get("n_estimators", 500),
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importance(importance_type="gain")
        
        # Calibrate probabilities if requested
        if self.calibrate and eval_set is not None:
            self._calibrate_probabilities(X, y)
        
        return self
    
    def _calibrate_probabilities(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """Apply isotonic calibration to improve probability estimates."""
        from sklearn.calibration import calibration_curve
        
        # Get raw predictions
        raw_probs = self.model.predict(X)
        
        # Simple isotonic calibration
        from sklearn.isotonic import IsotonicRegression
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_probs, y)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability scores.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of anomaly probabilities [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        raw_probs = self.model.predict(X)
        
        if self.calibrator is not None:
            return self.calibrator.transform(raw_probs)
        
        return raw_probs
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions.
        
        Args:
            X: Feature DataFrame
            threshold: Classification threshold
            
        Returns:
            Array of binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def score_with_explanation(
        self, 
        X: pd.DataFrame,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Score with per-prediction feature explanations.
        
        Args:
            X: Feature DataFrame
            top_k: Number of top features to return
            
        Returns:
            DataFrame with scores and explanations
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        probs = self.predict_proba(X)
        
        # For tree models, we can use feature contributions
        # This is a simplified version - for full SHAP, use shap library
        contributions = self._get_feature_contributions(X)
        
        results = []
        for i in range(len(X)):
            row_contributions = contributions[i]
            
            # Sort by absolute contribution
            sorted_indices = np.argsort(np.abs(row_contributions))[::-1][:top_k]
            
            top_features = [
                {
                    "feature": self.feature_names[idx],
                    "value": float(X.iloc[i, idx]),
                    "contribution": float(row_contributions[idx]),
                }
                for idx in sorted_indices
            ]
            
            results.append({
                "anomaly_score": float(probs[i]),
                "top_features": top_features,
            })
        
        return pd.DataFrame(results, index=X.index)
    
    def _get_feature_contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Get per-feature contributions for each prediction.
        
        Uses LightGBM's predict with pred_contrib=True.
        """
        # pred_contrib returns (n_samples, n_features + 1) 
        # where last column is the base value
        contributions = self.model.predict(X, pred_contrib=True)
        
        # Remove base value column
        return contributions[:, :-1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.feature_importances_,
        })
        
        return importance_df.sort_values("importance", ascending=False)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y: True labels
            
        Returns:
            Dict of evaluation metrics
        """
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        
        metrics = {
            "roc_auc": roc_auc_score(y, probs),
            "pr_auc": average_precision_score(y, probs),
            "accuracy": (preds == y).mean(),
        }
        
        # Compute FPR at different TPR thresholds
        precisions, recalls, thresholds = precision_recall_curve(y, probs)
        
        # Find threshold for 95% recall
        for i, recall in enumerate(recalls):
            if recall <= 0.95:
                threshold_95 = thresholds[i] if i < len(thresholds) else 0.5
                preds_95 = (probs >= threshold_95).astype(int)
                fpr_95 = ((preds_95 == 1) & (y == 0)).sum() / (y == 0).sum()
                metrics["fpr_at_95_recall"] = fpr_95
                break
        
        return metrics
    
    def save(self, path: Path):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Call fit() first.")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(path / "model.txt"))
        
        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "params": self.params,
            "calibrate": self.calibrate,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "LightGBMScorer":
        """Load model from disk."""
        path = Path(path)
        
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        
        scorer = cls(
            params=metadata["params"],
            calibrate=metadata["calibrate"],
        )
        scorer.feature_names = metadata["feature_names"]
        scorer.model = lgb.Booster(model_file=str(path / "model.txt"))
        scorer.feature_importances_ = scorer.model.feature_importance(importance_type="gain")
        
        return scorer
