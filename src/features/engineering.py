"""Main feature engineering pipeline for security events."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
from dataclasses import dataclass, field

from .windows import WindowAggregator, WindowConfig


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    windows: List[WindowConfig] = field(default_factory=list)
    selected_features: List[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "FeatureConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        
        windows = [
            WindowConfig(name=w["name"], duration_seconds=w["duration_seconds"])
            for w in config.get("windows", [])
        ]
        
        selected = []
        for group in config.get("selected_features", {}).values():
            selected.extend(group)
        
        return cls(windows=windows, selected_features=selected)


class FeatureEngineer:
    """Complete feature engineering pipeline for security events."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize feature engineer.
        
        Args:
            config: Feature configuration. If None, uses defaults.
        """
        if config is None:
            config = FeatureConfig(
                windows=[
                    WindowConfig("1min", 60),
                    WindowConfig("5min", 300),
                    WindowConfig("15min", 900),
                ]
            )
        
        self.config = config
        self.aggregator = WindowAggregator(config.windows)
        
        # Tracking for novelty detection
        self._seen_entities: Dict[str, set] = {
            "dst_ip": set(),
            "dst_port": set(),
            "user": set(),
        }
    
    def fit_transform(
        self,
        events: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Transform raw events into ML-ready features.
        
        Args:
            events: Raw security events DataFrame
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with engineered features, one row per window
        """
        # Step 1: Window aggregation
        window_features = self.aggregator.aggregate(events, timestamp_col)
        
        if window_features.empty:
            return pd.DataFrame()
        
        # Step 2: Rolling/baseline features
        window_features = self.aggregator.compute_rolling_features(
            window_features,
            lookback_windows=60  # 1 hour for 1-min windows
        )
        
        # Step 3: Novelty features
        window_features = self._add_novelty_features(events, window_features)
        
        # Step 4: Derived features
        window_features = self._add_derived_features(window_features)
        
        # Step 5: Fill missing values
        window_features = self._handle_missing(window_features)
        
        return window_features
    
    def transform(
        self,
        events: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Transform events using previously seen entities for novelty.
        
        Same as fit_transform but uses accumulated entity knowledge.
        """
        return self.fit_transform(events, timestamp_col)
    
    def _add_novelty_features(
        self,
        events: pd.DataFrame,
        window_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add features tracking never-before-seen entities."""
        
        events = events.copy()
        events["_window"] = events["timestamp"].dt.floor("60s")
        
        for window_start in window_features["window_start"]:
            window_events = events[events["_window"] == window_start]
            
            for entity_col in ["dst_ip", "dst_port"]:
                if entity_col not in window_events.columns:
                    continue
                
                current_entities = set(window_events[entity_col].dropna().unique())
                
                if len(current_entities) > 0:
                    new_entities = current_entities - self._seen_entities[entity_col]
                    new_ratio = len(new_entities) / len(current_entities)
                    
                    # Update window features
                    mask = window_features["window_start"] == window_start
                    window_features.loc[mask, f"new_{entity_col}_ratio_1min"] = new_ratio
                    window_features.loc[mask, f"new_{entity_col}_count_1min"] = len(new_entities)
                    
                    # Track these entities
                    self._seen_entities[entity_col].update(current_entities)
        
        return window_features
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived/combined features."""
        
        # Byte ratio (out/in)
        for suffix in ["1min", "5min", "15min"]:
            in_col = f"bytes_in_mean_{suffix}"
            out_col = f"bytes_out_mean_{suffix}"
            
            if in_col in df.columns and out_col in df.columns:
                df[f"bytes_ratio_{suffix}"] = (
                    df[out_col] / (df[in_col] + 1)  # Add 1 to avoid div by zero
                )
        
        # Port scan indicator
        for suffix in ["1min", "5min"]:
            ports_col = f"unique_dst_ports_{suffix}"
            ips_col = f"unique_dst_ips_{suffix}"
            
            if ports_col in df.columns and ips_col in df.columns:
                df[f"port_scan_indicator_{suffix}"] = (
                    (df[ports_col] > 50) & (df[ips_col] < 5)
                ).astype(int)
        
        # High entropy indicator
        for suffix in ["1min", "5min"]:
            entropy_col = f"port_entropy_{suffix}"
            if entropy_col in df.columns:
                df[f"high_entropy_{suffix}"] = (df[entropy_col] > 4.0).astype(int)
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        
        # Numeric columns: fill with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Ratio columns: clip to [0, 1]
        ratio_cols = [c for c in df.columns if "ratio" in c.lower()]
        for col in ratio_cols:
            if col in df.columns:
                df[col] = df[col].clip(0, 1)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names produced."""
        # This would be populated after a fit_transform call
        base_features = []
        
        for window in self.config.windows:
            suffix = window.name
            base_features.extend([
                f"event_count_{suffix}",
                f"unique_src_ips_{suffix}",
                f"unique_dst_ips_{suffix}",
                f"unique_dst_ports_{suffix}",
                f"port_entropy_{suffix}",
                f"src_ip_entropy_{suffix}",
                f"bytes_in_mean_{suffix}",
                f"bytes_out_mean_{suffix}",
                f"conn_duration_p50_{suffix}",
                f"conn_duration_p95_{suffix}",
                f"failed_auth_ratio_{suffix}",
                f"inter_arrival_cv_{suffix}",
                f"baseline_deviation_{suffix}",
                f"bytes_ratio_{suffix}",
            ])
        
        return base_features
    
    def reset_novelty_tracking(self):
        """Reset seen entities for novelty detection."""
        for key in self._seen_entities:
            self._seen_entities[key] = set()


def load_unsw_nb15(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load UNSW-NB15 dataset and return features + labels.
    
    Args:
        path: Path to UNSW-NB15 CSV file
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    # UNSW-NB15 has pre-computed features
    df = pd.read_csv(path)
    
    # Map column names to our schema
    column_mapping = {
        "srcip": "src_ip",
        "dstip": "dst_ip",
        "sport": "src_port",
        "dsport": "dst_port",
        "proto": "protocol",
        "dur": "duration",
        "sbytes": "bytes_out",
        "dbytes": "bytes_in",
        "sttl": "src_ttl",
        "dttl": "dst_ttl",
        "attack_cat": "attack_category",
        "label": "is_attack",
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Extract labels
    labels = df["is_attack"] if "is_attack" in df.columns else None
    
    # Feature columns (numeric only, excluding labels)
    feature_cols = [
        c for c in df.columns 
        if c not in ["is_attack", "attack_category", "src_ip", "dst_ip"]
        and df[c].dtype in [np.float64, np.int64, float, int]
    ]
    
    features = df[feature_cols]
    
    return features, labels
