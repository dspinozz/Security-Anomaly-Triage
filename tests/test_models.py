"""Model training and inference tests."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline import BaselineScorer, RuleConfig
from src.models.lightgbm_model import LightGBMScorer
from src.models.isolation import IsolationForestScorer


class TestBaselineScorer:
    """Tests for rule-based baseline scorer."""
    
    def test_baseline_scorer_creation(self):
        scorer = BaselineScorer()
        assert scorer.config is not None
    
    def test_baseline_scorer_with_custom_config(self):
        config = RuleConfig(alert_threshold=0.5)
        scorer = BaselineScorer(config=config)
        assert scorer.config.alert_threshold == 0.5
    
    def test_baseline_score_returns_dataframe(self):
        scorer = BaselineScorer()
        
        # Sample features
        features = pd.DataFrame({
            'event_count_1min': [10, 100, 500],
            'unique_dst_ports_1min': [5, 50, 200],
            'bytes_out_mean_1min': [1000, 50000, 1000000],
        })
        
        result = scorer.score(features)
        
        assert isinstance(result, pd.DataFrame)
        assert 'baseline_score' in result.columns
    
    def test_baseline_scores_are_bounded(self):
        scorer = BaselineScorer()
        
        features = pd.DataFrame({
            'event_count_1min': np.random.randint(1, 1000, 100),
            'unique_dst_ports_1min': np.random.randint(1, 100, 100),
        })
        
        result = scorer.score(features)
        
        # Scores should be between 0 and 1
        assert (result['baseline_score'] >= 0).all()
        assert (result['baseline_score'] <= 1).all()


class TestLightGBMScorer:
    """Tests for LightGBM classifier."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 500
        
        # Create features that correlate with labels
        X = pd.DataFrame({
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
            'feature3': np.random.randn(n),
        })
        
        # Labels correlated with feature1
        y = (X['feature1'] + np.random.randn(n) * 0.5 > 0).astype(int)
        
        return X, y
    
    def test_lightgbm_scorer_creation(self):
        scorer = LightGBMScorer()
        assert scorer.model is None  # Not fitted yet
    
    def test_lightgbm_fit(self, sample_data):
        X, y = sample_data
        scorer = LightGBMScorer()
        
        scorer.fit(X, y)
        
        assert scorer.model is not None
    
    def test_lightgbm_predict_proba(self, sample_data):
        X, y = sample_data
        scorer = LightGBMScorer()
        scorer.fit(X, y)
        
        probs = scorer.predict_proba(X)
        
        assert len(probs) == len(X)
        assert all(0 <= p <= 1 for p in probs)
    
    def test_lightgbm_evaluate(self, sample_data):
        X, y = sample_data
        X_train, X_test = X[:400], X[400:]
        y_train, y_test = y[:400], y[400:]
        
        scorer = LightGBMScorer()
        scorer.fit(X_train, y_train)
        
        metrics = scorer.evaluate(X_test, y_test)
        
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_lightgbm_feature_importance(self, sample_data):
        X, y = sample_data
        scorer = LightGBMScorer()
        scorer.fit(X, y)
        
        importance = scorer.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns


class TestIsolationForestScorer:
    """Tests for Isolation Forest unsupervised anomaly detector."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with anomalies."""
        np.random.seed(42)
        
        # Normal data
        normal = np.random.randn(200, 3)
        
        # Anomalies (outliers)
        anomalies = np.random.randn(20, 3) * 5 + 10
        
        X = pd.DataFrame(
            np.vstack([normal, anomalies]),
            columns=['f1', 'f2', 'f3']
        )
        
        return X
    
    def test_isolation_forest_creation(self):
        scorer = IsolationForestScorer()
        assert scorer is not None
    
    def test_isolation_forest_fit(self, sample_data):
        scorer = IsolationForestScorer(contamination=0.1)
        scorer.fit(sample_data)
        
        # Should be fitted
        assert scorer.isolation_forest is not None
    
    def test_isolation_forest_score(self, sample_data):
        scorer = IsolationForestScorer(contamination=0.1)
        scorer.fit(sample_data)
        
        result = scorer.score(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'isolation_score' in result.columns or 'combined_score' in result.columns


class TestModelSaveLoad:
    """Tests for model persistence."""
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return tmp_path
    
    def test_lightgbm_save_load(self, temp_dir):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        y = (X['a'] > 0).astype(int)
        
        # Train and save
        scorer = LightGBMScorer()
        scorer.fit(X, y)
        scorer.save(temp_dir / "lgb")
        
        # Load and predict
        loaded = LightGBMScorer.load(temp_dir / "lgb")
        preds = loaded.predict_proba(X)
        
        assert len(preds) == len(X)


# Run with: pytest tests/test_models.py -v
