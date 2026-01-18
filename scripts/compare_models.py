#!/usr/bin/env python3
"""Compare LightGBM, XGBoost, and CatBoost on security anomaly detection.

This script provides a fair comparison of gradient boosting frameworks
for tabular security data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.table import Table

console = Console()

# ============================================================
# Model Implementations
# ============================================================

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM classifier."""
    import lightgbm as lgb
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_samples": 20,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params["scale_pos_weight"] = scale_pos_weight
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
    ]
    
    start = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )
    train_time = time.time() - start
    
    return model, train_time, lambda X: model.predict(X)


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier."""
    import xgboost as xgb
    
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    start = time.time()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    train_time = time.time() - start
    
    return model, train_time, lambda X: model.predict(xgb.DMatrix(X))


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost classifier."""
    from catboost import CatBoostClassifier
    
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos_weight,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
        thread_count=-1,
    )
    
    start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )
    train_time = time.time() - start
    
    return model, train_time, lambda X: model.predict_proba(X)[:, 1]


# ============================================================
# Data Loading (Event-Level for Fair Comparison)
# ============================================================

def load_event_level_data(path: Path):
    """Load data at event level (not window-aggregated).
    
    This gives us more samples and avoids window-label alignment issues.
    """
    df = pd.read_csv(path)
    
    # Drop non-numeric and ID columns
    drop_cols = ['timestamp', 'src_ip', 'dst_ip', 'event_type', 'status', 
                 'user', 'protocol', 'attack_category']
    
    # Keep only numeric features
    feature_cols = ['src_port', 'dst_port', 'bytes_in', 'bytes_out', 'duration']
    
    # Encode categorical if needed
    if 'protocol' in df.columns:
        protocol_dummies = pd.get_dummies(df['protocol'], prefix='proto')
        df = pd.concat([df, protocol_dummies], axis=1)
        feature_cols.extend(protocol_dummies.columns.tolist())
    
    if 'status' in df.columns:
        df['status_failed'] = (df['status'] == 'failed').astype(int)
        feature_cols.append('status_failed')
    
    # Ensure features exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    return X, y


# ============================================================
# Main Comparison
# ============================================================

def run_comparison(data_path: Path):
    """Run full comparison of all three models."""
    
    console.print("\n[bold blue]=" * 60)
    console.print("[bold blue]GRADIENT BOOSTING MODEL COMPARISON")
    console.print("[bold blue]=" * 60)
    
    # Load data
    console.print("\n[yellow]Loading data...[/yellow]")
    X, y = load_event_level_data(data_path)
    
    console.print(f"  Samples: {len(X):,}")
    console.print(f"  Features: {len(X.columns)}")
    console.print(f"  Positive rate: {y.mean()*100:.1f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    console.print(f"\n  Train: {len(X_train):,}")
    console.print(f"  Val: {len(X_val):,}")
    console.print(f"  Test: {len(X_test):,}")
    
    # Train and evaluate each model
    results = {}
    
    models = [
        ("LightGBM", train_lightgbm),
        ("XGBoost", train_xgboost),
        ("CatBoost", train_catboost),
    ]
    
    for name, train_fn in models:
        console.print(f"\n[bold]Training {name}...[/bold]")
        
        try:
            model, train_time, predict_fn = train_fn(
                X_train.values, y_train.values,
                X_val.values, y_val.values
            )
            
            # Evaluate
            y_pred_proba = predict_fn(X_test.values)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            results[name] = {
                "train_time": train_time,
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
                "pr_auc": average_precision_score(y_test, y_pred_proba),
                "f1": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
            }
            
            console.print(f"  ✓ Completed in {train_time:.2f}s")
            
        except Exception as e:
            console.print(f"  ✗ Failed: {e}")
            results[name] = None
    
    # Display results
    console.print("\n" + "=" * 60)
    console.print("[bold]RESULTS COMPARISON[/bold]")
    console.print("=" * 60)
    
    table = Table(title="Model Performance Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("ROC-AUC", style="green")
    table.add_column("PR-AUC", style="green")
    table.add_column("F1", style="yellow")
    table.add_column("Precision", style="blue")
    table.add_column("Recall", style="blue")
    table.add_column("Train Time", style="magenta")
    
    for name, metrics in results.items():
        if metrics:
            table.add_row(
                name,
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['pr_auc']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['train_time']:.2f}s",
            )
        else:
            table.add_row(name, "FAILED", "-", "-", "-", "-", "-")
    
    console.print(table)
    
    # Recommendation
    console.print("\n" + "=" * 60)
    console.print("[bold]ANALYSIS & RECOMMENDATION[/bold]")
    console.print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if v}
    
    if valid_results:
        # Find best by PR-AUC (most important for imbalanced data)
        best_pr = max(valid_results.items(), key=lambda x: x[1]['pr_auc'])
        # Find fastest
        fastest = min(valid_results.items(), key=lambda x: x[1]['train_time'])
        # Find best ROC-AUC
        best_roc = max(valid_results.items(), key=lambda x: x[1]['roc_auc'])
        
        console.print(f"""
[green]Best PR-AUC (imbalanced data):[/green] {best_pr[0]} ({best_pr[1]['pr_auc']:.4f})
[green]Best ROC-AUC:[/green] {best_roc[0]} ({best_roc[1]['roc_auc']:.4f})
[green]Fastest Training:[/green] {fastest[0]} ({fastest[1]['train_time']:.2f}s)
""")

        console.print("""
[bold]When to use each:[/bold]

[cyan]LightGBM[/cyan]
  ✓ Best for: Large datasets, fast training, low memory
  ✓ Strengths: Speed, leaf-wise growth, GPU support
  ✓ Use when: Training time matters, millions of rows

[cyan]XGBoost[/cyan]
  ✓ Best for: General purpose, good default behavior
  ✓ Strengths: Regularization, handling missing values
  ✓ Use when: Need battle-tested, well-documented solution

[cyan]CatBoost[/cyan]
  ✓ Best for: Categorical features, preventing overfitting
  ✓ Strengths: Ordered boosting, native categorical handling
  ✓ Use when: Many categorical features, less tuning desired

[bold yellow]For Security Data Recommendation:[/bold yellow]
  → LightGBM is typically best for security log data because:
    1. Security logs are often massive (millions of events)
    2. Features are mostly numeric after encoding
    3. Fast inference is critical for real-time alerting
    4. Memory efficiency matters for production deployment
""")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare gradient boosting models")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/unsw-nb15/UNSW_NB15_training-set.csv"),
        help="Path to data CSV"
    )
    args = parser.parse_args()
    
    if not args.data.exists():
        console.print(f"[red]Data file not found: {args.data}[/red]")
        return
    
    run_comparison(args.data)


if __name__ == "__main__":
    main()
