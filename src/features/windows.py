"""Time window aggregation for security events."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import entropy
from dataclasses import dataclass


@dataclass
class WindowConfig:
    """Configuration for a time window."""
    name: str
    duration_seconds: int


class WindowAggregator:
    """Aggregate security events into fixed time windows with features."""
    
    def __init__(self, window_configs: List[WindowConfig]):
        """Initialize with window configurations.
        
        Args:
            window_configs: List of window configurations (e.g., 1min, 5min, 15min)
        """
        self.windows = window_configs
    
    def aggregate(
        self,
        events: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Aggregate events into windows with computed features.
        
        Args:
            events: DataFrame with security events
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with one row per window, containing aggregated features
        """
        if events.empty:
            return pd.DataFrame()
        
        # Ensure timestamp is datetime
        events = events.copy()
        events[timestamp_col] = pd.to_datetime(events[timestamp_col])
        events = events.sort_values(timestamp_col)
        
        all_features = []
        
        for window in self.windows:
            # Create window boundaries
            window_features = self._aggregate_window(
                events, 
                window.duration_seconds,
                window.name,
                timestamp_col
            )
            all_features.append(window_features)
        
        # Merge all window sizes
        if not all_features:
            return pd.DataFrame()
        
        result = all_features[0]
        for wf in all_features[1:]:
            result = result.merge(
                wf, 
                on="window_start", 
                how="outer",
                suffixes=("", "_dup")
            )
            # Remove duplicate columns
            result = result.loc[:, ~result.columns.str.endswith("_dup")]
        
        return result.sort_values("window_start")
    
    def _aggregate_window(
        self,
        events: pd.DataFrame,
        duration_seconds: int,
        window_name: str,
        timestamp_col: str,
    ) -> pd.DataFrame:
        """Aggregate events for a specific window duration."""
        
        # Create window key
        events["_window_start"] = events[timestamp_col].dt.floor(f"{duration_seconds}s")
        
        features = []
        
        for window_start, window_events in events.groupby("_window_start"):
            row = {"window_start": window_start}
            
            # Count features
            row[f"event_count_{window_name}"] = len(window_events)
            
            # Cardinality features
            for col in ["src_ip", "dst_ip", "dst_port", "user", "protocol"]:
                if col in window_events.columns:
                    row[f"unique_{col}s_{window_name}"] = window_events[col].nunique()
            
            # Entropy features
            if "dst_port" in window_events.columns:
                row[f"port_entropy_{window_name}"] = self._compute_entropy(
                    window_events["dst_port"]
                )
            if "src_ip" in window_events.columns:
                row[f"src_ip_entropy_{window_name}"] = self._compute_entropy(
                    window_events["src_ip"]
                )
            
            # Byte statistics
            for col in ["bytes_in", "bytes_out"]:
                if col in window_events.columns:
                    vals = window_events[col].dropna()
                    if len(vals) > 0:
                        row[f"{col}_mean_{window_name}"] = vals.mean()
                        row[f"{col}_std_{window_name}"] = vals.std() if len(vals) > 1 else 0
                        row[f"{col}_sum_{window_name}"] = vals.sum()
            
            # Duration statistics
            if "duration" in window_events.columns:
                durations = window_events["duration"].dropna()
                if len(durations) > 0:
                    row[f"conn_duration_p50_{window_name}"] = durations.quantile(0.5)
                    row[f"conn_duration_p95_{window_name}"] = durations.quantile(0.95)
            
            # Auth failure ratio
            if "status" in window_events.columns and "event_type" in window_events.columns:
                auth_events = window_events[window_events["event_type"] == "auth"]
                if len(auth_events) > 0:
                    failed = (auth_events["status"] == "failed").sum()
                    row[f"failed_auth_ratio_{window_name}"] = failed / len(auth_events)
                else:
                    row[f"failed_auth_ratio_{window_name}"] = 0.0
            
            # Inter-arrival time statistics (burstiness)
            if len(window_events) > 1:
                times = window_events[timestamp_col].sort_values()
                inter_arrival = times.diff().dt.total_seconds().dropna()
                if len(inter_arrival) > 0 and inter_arrival.mean() > 0:
                    row[f"inter_arrival_mean_{window_name}"] = inter_arrival.mean()
                    row[f"inter_arrival_cv_{window_name}"] = (
                        inter_arrival.std() / inter_arrival.mean()
                        if inter_arrival.mean() > 0 else 0
                    )
            
            # Protocol distribution
            if "protocol" in window_events.columns:
                for proto in ["http", "https", "dns", "ssh", "smtp"]:
                    proto_count = (window_events["protocol"] == proto).sum()
                    row[f"{proto}_ratio_{window_name}"] = proto_count / len(window_events)
            
            features.append(row)
        
        events.drop("_window_start", axis=1, inplace=True)
        return pd.DataFrame(features)
    
    @staticmethod
    def _compute_entropy(series: pd.Series) -> float:
        """Compute Shannon entropy of a categorical series."""
        if len(series) == 0:
            return 0.0
        value_counts = series.value_counts(normalize=True)
        return float(entropy(value_counts, base=2))
    
    def compute_rolling_features(
        self,
        window_features: pd.DataFrame,
        lookback_windows: int = 60,
    ) -> pd.DataFrame:
        """Add rolling/historical features.
        
        Args:
            window_features: DataFrame from aggregate()
            lookback_windows: Number of windows to look back
            
        Returns:
            DataFrame with additional rolling features
        """
        df = window_features.copy()
        
        # Find event count columns
        count_cols = [c for c in df.columns if c.startswith("event_count_")]
        
        for col in count_cols:
            suffix = col.replace("event_count_", "")
            
            # Rolling mean (baseline)
            df[f"baseline_mean_{suffix}"] = (
                df[col]
                .rolling(window=lookback_windows, min_periods=1)
                .mean()
            )
            
            # Rolling std
            df[f"baseline_std_{suffix}"] = (
                df[col]
                .rolling(window=lookback_windows, min_periods=2)
                .std()
                .fillna(1.0)  # Avoid division by zero
            )
            
            # Z-score deviation from baseline
            df[f"baseline_deviation_{suffix}"] = (
                (df[col] - df[f"baseline_mean_{suffix}"]) / 
                df[f"baseline_std_{suffix}"].clip(lower=0.1)
            )
            
            # Rate of change
            df[f"rate_change_{suffix}"] = df[col].pct_change(fill_method=None).fillna(0)
        
        return df
