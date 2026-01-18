#!/usr/bin/env python3
"""Training script for security anomaly models.

Usage:
    python scripts/train.py --data data/synthetic/synthetic_events.csv
    python scripts/train.py --data data/unsw-nb15/UNSW_NB15_training-set.csv --dataset unsw-nb15
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineering import FeatureEngineer, FeatureConfig
from src.features.windows import WindowConfig
from src.models.lightgbm_model import LightGBMScorer
from src.models.isolation import IsolationForestScorer
from src.models.baseline import BaselineScorer
from src.scoring.scorer import UnifiedScorer


console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "features.yaml"


def load_synthetic_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load synthetic dataset."""
    df = pd.read_csv(path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract labels
    labels = df["label"]
    
    # Drop non-feature columns
    drop_cols = ["label", "attack_category", "timestamp", "event_type", "status"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Need to aggregate into windows first
    console.print("[yellow]Aggregating events into windows...[/yellow]")
    
    # Convert to features per window
    engineer = FeatureEngineer()
    features = engineer.fit_transform(df)
    
    # For labels, we'll use majority class per window
    # This is simplified - in practice you'd align windows with labels
    df["_window"] = pd.to_datetime(df["timestamp"]).dt.floor("60s")
    window_labels = df.groupby("_window")["label"].apply(lambda x: 1 if x.sum() > 0 else 0)
    
    # Align features with labels
    features = features.set_index("window_start")
    aligned_labels = features.index.map(lambda x: window_labels.get(x, 0))
    
    return features, pd.Series(aligned_labels, index=features.index)


def load_unsw_nb15_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load UNSW-NB15 dataset (pre-computed features)."""
    df = pd.read_csv(path)
    
    # Standard UNSW-NB15 column names
    label_col = "Label" if "Label" in df.columns else "label"
    
    if label_col not in df.columns:
        raise ValueError(f"Label column not found. Available columns: {df.columns.tolist()}")
    
    labels = df[label_col].astype(int)
    
    # Drop non-numeric and label columns
    drop_cols = [label_col, "attack_cat", "srcip", "dstip", "proto", "state", "service"]
    feature_cols = [
        c for c in df.columns 
        if c not in drop_cols and df[c].dtype in [np.float64, np.int64, float, int]
    ]
    
    features = df[feature_cols].fillna(0)
    
    return features, labels


def train(
    data_path: Path,
    dataset_type: str = "synthetic",
    output_dir: Path = None,
    test_size: float = 0.2,
):
    """Train all models on dataset."""
    
    output_dir = output_dir or PROJECT_ROOT / "models" / "trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold blue]Training Security Anomaly Models[/bold blue]")
    console.print(f"Dataset: {data_path}")
    console.print(f"Type: {dataset_type}")
    console.print(f"Output: {output_dir}\n")
    
    # Load data
    console.print("[yellow]Loading data...[/yellow]")
    if dataset_type == "synthetic":
        features, labels = load_synthetic_data(data_path)
    elif dataset_type == "unsw-nb15":
        features, labels = load_unsw_nb15_data(data_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    console.print(f"Loaded {len(features)} samples, {labels.sum()} attacks ({100*labels.mean():.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=test_size, 
        stratify=labels,
        random_state=42
    )
    
    console.print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 1. Train LightGBM (supervised)
    console.print("\n[bold]Training LightGBM classifier...[/bold]")
    lgb_scorer = LightGBMScorer(calibrate=True)
    lgb_scorer.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )
    
    lgb_metrics = lgb_scorer.evaluate(X_test, y_test)
    console.print(f"  ROC-AUC: {lgb_metrics['roc_auc']:.4f}")
    console.print(f"  PR-AUC: {lgb_metrics['pr_auc']:.4f}")
    console.print(f"  FPR@95: {lgb_metrics.get('fpr_at_95_recall', 'N/A')}")
    
    lgb_scorer.save(output_dir / "lgb")
    
    # 2. Train Isolation Forest (unsupervised)
    console.print("\n[bold]Training Isolation Forest...[/bold]")
    if_scorer = IsolationForestScorer(
        contamination=0.1,
        n_estimators=100,
        use_lof=True
    )
    if_scorer.fit(X_train)
    
    # Evaluate IF on test set
    if_scores = if_scorer.score(X_test)
    from sklearn.metrics import roc_auc_score, average_precision_score
    if_auc = roc_auc_score(y_test, if_scores["combined_score"])
    if_pr = average_precision_score(y_test, if_scores["combined_score"])
    
    console.print(f"  ROC-AUC: {if_auc:.4f}")
    console.print(f"  PR-AUC: {if_pr:.4f}")
    
    if_scorer.save(output_dir / "isolation")
    
    # 3. Create unified scorer
    console.print("\n[bold]Creating unified scorer...[/bold]")
    baseline_scorer = BaselineScorer()
    
    unified = UnifiedScorer(
        baseline_scorer=baseline_scorer,
        lgb_scorer=lgb_scorer,
        if_scorer=if_scorer,
        weights={"baseline": 0.2, "lgb": 0.5, "isolation": 0.3}
    )
    unified.save(output_dir)
    
    # 4. Compute final metrics
    console.print("\n[bold]Final Evaluation[/bold]")
    scored = unified.score(X_test, return_explanations=False)
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    final_auc = roc_auc_score(y_test, scored["anomaly_score"])
    final_pr = average_precision_score(y_test, scored["anomaly_score"])
    
    # Alert reduction
    reduction = unified.compute_alert_reduction(
        X_test,
        baseline_threshold=0.0,
        model_threshold=0.5
    )
    
    # Print results table
    table = Table(title="Model Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Unified ROC-AUC", f"{final_auc:.4f}")
    table.add_row("Unified PR-AUC", f"{final_pr:.4f}")
    table.add_row("LightGBM ROC-AUC", f"{lgb_metrics['roc_auc']:.4f}")
    table.add_row("Isolation Forest ROC-AUC", f"{if_auc:.4f}")
    table.add_row("Alert Reduction Rate", f"{reduction['alert_reduction_rate']:.1%}")
    table.add_row("Model Alerts", str(reduction["model_alerts"]))
    table.add_row("Total Events", str(reduction["total_events"]))
    
    console.print(table)
    
    # Save metrics
    metrics = {
        "unified_roc_auc": final_auc,
        "unified_pr_auc": final_pr,
        "lgb_roc_auc": lgb_metrics["roc_auc"],
        "lgb_pr_auc": lgb_metrics["pr_auc"],
        "if_roc_auc": if_auc,
        "if_pr_auc": if_pr,
        "alert_reduction_rate": reduction["alert_reduction_rate"],
        "test_size": len(X_test),
        "train_size": len(X_train),
        "attack_rate": float(labels.mean()),
    }
    
    with open(output_dir.parent / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Feature importance
    console.print("\n[bold]Top 10 Feature Importance[/bold]")
    importance = lgb_scorer.get_feature_importance()
    for i, row in importance.head(10).iterrows():
        console.print(f"  {row['feature']}: {row['importance']:.2f}")
    
    importance.to_csv(output_dir.parent / "feature_importance.csv", index=False)
    
    console.print(f"\n[green]✓ Training complete![/green]")
    console.print(f"Models saved to: {output_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train security anomaly models")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "unsw-nb15"],
        default="synthetic",
        help="Dataset type"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for models"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction"
    )
    
    args = parser.parse_args()
    
    if not args.data.exists():
        console.print(f"[red]Error: Data file not found: {args.data}[/red]")
        console.print("Run: python scripts/download_data.py --dataset synthetic")
        return
    
    train(
        data_path=args.data,
        dataset_type=args.dataset,
        output_dir=args.output,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
