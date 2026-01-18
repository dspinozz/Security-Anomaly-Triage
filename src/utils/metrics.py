"""Evaluation metrics for security anomaly detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        y_pred: Optional binary predictions (computed from threshold if None)
        threshold: Classification threshold
        
    Returns:
        Dict of metric name -> value
    """
    if y_pred is None:
        y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {}
    
    # AUC metrics
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
    except ValueError:
        metrics["roc_auc"] = 0.0
    
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_scores)
    except ValueError:
        metrics["pr_auc"] = 0.0
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Detection metrics
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)
    
    # FPR at specific recall thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    for target_recall in [0.9, 0.95, 0.99]:
        fpr_at_recall = compute_fpr_at_recall(y_true, y_scores, target_recall)
        metrics[f"fpr_at_{int(target_recall*100)}_recall"] = fpr_at_recall
    
    return metrics


def compute_fpr_at_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_recall: float,
) -> float:
    """Compute FPR at a specific recall threshold.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        target_recall: Target recall level
        
    Returns:
        False positive rate at the threshold achieving target recall
    """
    # Sort by score descending
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    
    total_positives = y_true.sum()
    total_negatives = len(y_true) - total_positives
    
    if total_positives == 0 or total_negatives == 0:
        return 0.0
    
    target_tp = int(target_recall * total_positives)
    
    # Find threshold that achieves target recall
    tp_count = 0
    fp_count = 0
    
    for i, is_positive in enumerate(y_true_sorted):
        if is_positive:
            tp_count += 1
        else:
            fp_count += 1
        
        if tp_count >= target_tp:
            break
    
    return fp_count / total_negatives


def compute_alert_reduction(
    baseline_alerts: int,
    model_alerts: int,
    total_events: int,
) -> Dict[str, float]:
    """Compute alert reduction metrics.
    
    Args:
        baseline_alerts: Number of alerts from baseline (or all events)
        model_alerts: Number of alerts from model
        total_events: Total number of events
        
    Returns:
        Dict of reduction metrics
    """
    return {
        "baseline_alerts": baseline_alerts,
        "model_alerts": model_alerts,
        "absolute_reduction": baseline_alerts - model_alerts,
        "relative_reduction": 1.0 - (model_alerts / max(baseline_alerts, 1)),
        "alert_rate": model_alerts / max(total_events, 1),
        "alerts_per_1000": (model_alerts / max(total_events, 1)) * 1000,
    }


def compute_detection_latency(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute detection latency metrics.
    
    Args:
        timestamps: Event/window timestamps
        y_true: True labels
        y_scores: Predicted scores
        threshold: Detection threshold
        
    Returns:
        Dict of latency metrics
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    # Find true attack start times and detection times
    latencies = []
    
    # Group consecutive attacks
    attack_starts = []
    attack_detected = []
    in_attack = False
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == 1 and not in_attack:
            # Attack starts
            in_attack = True
            attack_starts.append(i)
            attack_detected.append(None)
        
        if in_attack and pred == 1 and attack_detected[-1] is None:
            # First detection of this attack
            attack_detected[-1] = i
        
        if true == 0:
            in_attack = False
    
    # Compute latencies (in number of windows)
    for start, detected in zip(attack_starts, attack_detected):
        if detected is not None:
            latencies.append(detected - start)
    
    if not latencies:
        return {
            "mean_latency_windows": 0,
            "median_latency_windows": 0,
            "p95_latency_windows": 0,
            "detection_rate": 0.0,
        }
    
    return {
        "mean_latency_windows": np.mean(latencies),
        "median_latency_windows": np.median(latencies),
        "p95_latency_windows": np.percentile(latencies, 95),
        "detection_rate": len(latencies) / len(attack_starts),
        "attacks_detected": len(latencies),
        "attacks_total": len(attack_starts),
    }


def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics"):
    """Print metrics in a formatted table."""
    try:
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for name, value in metrics.items():
            if isinstance(value, float):
                if value < 1 and value > 0:
                    table.add_row(name, f"{value:.4f}")
                else:
                    table.add_row(name, f"{value:.2f}")
            else:
                table.add_row(name, str(value))
        
        console.print(table)
    
    except ImportError:
        print(f"\n{title}")
        print("=" * 40)
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")
