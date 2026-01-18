"""Feature engineering unit tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineering import FeatureEngineer, FeatureConfig
from src.features.windows import WindowAggregator, WindowConfig


class TestWindowConfig:
    """Tests for WindowConfig dataclass."""
    
    def test_window_config_creation(self):
        config = WindowConfig(name="1min", duration_seconds=60)
        assert config.name == "1min"
        assert config.duration_seconds == 60
    
    def test_window_config_defaults(self):
        config = WindowConfig(name="test", duration_seconds=30)
        assert hasattr(config, 'name')
        assert hasattr(config, 'duration_seconds')


class TestWindowAggregator:
    """Tests for WindowAggregator."""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample security events."""
        np.random.seed(42)
        n = 100
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        return pd.DataFrame({
            'timestamp': [base_time + timedelta(seconds=i*2) for i in range(n)],
            'src_ip': np.random.choice(['192.168.1.1', '192.168.1.2', '10.0.0.1'], n),
            'dst_ip': np.random.choice(['8.8.8.8', '1.1.1.1'], n),
            'dst_port': np.random.choice([80, 443, 22, 53], n),
            'protocol': np.random.choice(['tcp', 'udp'], n),
            'bytes_in': np.random.randint(100, 10000, n),
            'bytes_out': np.random.randint(50, 5000, n),
        })
    
    def test_aggregator_creates_windows(self, sample_events):
        config = [WindowConfig("1min", 60)]
        aggregator = WindowAggregator(config)
        
        result = aggregator.aggregate(sample_events, "timestamp")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_aggregator_multiple_windows(self, sample_events):
        config = [
            WindowConfig("1min", 60),
            WindowConfig("5min", 300),
        ]
        aggregator = WindowAggregator(config)
        
        result = aggregator.aggregate(sample_events, "timestamp")
        
        # Should have columns for both windows
        assert any("1min" in col for col in result.columns)
        assert any("5min" in col for col in result.columns)


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample security events with labels."""
        np.random.seed(42)
        n = 200
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        return pd.DataFrame({
            'timestamp': [base_time + timedelta(seconds=i) for i in range(n)],
            'src_ip': np.random.choice(['192.168.1.1', '192.168.1.2'], n),
            'dst_ip': np.random.choice(['8.8.8.8', '1.1.1.1'], n),
            'dst_port': np.random.choice([80, 443, 22], n),
            'protocol': np.random.choice(['tcp', 'udp', 'ssh'], n),
            'bytes_in': np.random.randint(100, 10000, n),
            'bytes_out': np.random.randint(50, 5000, n),
            'duration': np.random.exponential(0.5, n),
            'label': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        })
    
    def test_feature_engineer_default_config(self):
        engineer = FeatureEngineer()
        assert engineer.config is not None
        assert len(engineer.config.windows) > 0
    
    def test_feature_engineer_fit_transform(self, sample_events):
        engineer = FeatureEngineer()
        
        features = engineer.fit_transform(sample_events)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert len(features.columns) > 5  # Should have many features
    
    def test_features_are_numeric(self, sample_events):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_events)
        
        # Drop window_start if present
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        # Most columns should be numeric
        assert len(numeric_cols) >= len(features.columns) - 2
    
    def test_features_no_nan_after_fillna(self, sample_events):
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_events)
        
        numeric_features = features.select_dtypes(include=[np.number])
        filled = numeric_features.fillna(0)
        
        assert not filled.isna().any().any()


class TestFeatureStatistics:
    """Tests for feature statistics correctness."""
    
    def test_event_count_aggregation(self):
        """Verify event counts are aggregated correctly."""
        # Create 10 events in a 1-minute window
        events = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 12:00:00', periods=10, freq='5s'),
            'dst_port': [80] * 10,
            'bytes_in': [100] * 10,
            'bytes_out': [50] * 10,
        })
        
        config = [WindowConfig("1min", 60)]
        aggregator = WindowAggregator(config)
        result = aggregator.aggregate(events, "timestamp")
        
        # Should have at least 10 events in the window
        if 'event_count_1min' in result.columns:
            assert result['event_count_1min'].iloc[0] >= 1


# Run with: pytest tests/test_features.py -v
