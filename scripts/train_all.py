#!/usr/bin/env python3
"""Extended training script that saves LightGBM, XGBoost, and CatBoost models.

Usage:
    python scripts/train_all.py --data data/synthetic/synthetic_events.csv
"""

import argparse
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from rich.console import Console
from rich.table import Table
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()
PROJECT_ROOT = Path(__file__).parent.parent


def load_event_level_data(path: Path):
    """Load data at event level for multi-model training."""
    df = pd.read_csv(path)
    
    # Create features
    feature_cols = ['src_port', 'dst_port', 'bytes_in', 'bytes_out', 'duration']
    
    # Encode categorical
    if 'protocol' in df.columns:
        protocol_dummies = pd.get_dummies(df['protocol'], prefix='proto')
        df = pd.concat([df, protocol_dummies], axis=1)
        feature_cols.extend(protocol_dummies.columns.tolist())
    
    if 'status' in df.columns:
        df['status_failed'] = (df['status'] == 'failed').astype(int)
        feature_cols.append('status_failed')
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    return X, y, feature_cols


def train_all_models(data_path: Path, output_dir: Path = None):
    """Train LightGBM, XGBoost, and CatBoost models."""
    
    output_dir = output_dir or PROJECT_ROOT / "models" / "trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("\n[bold blue]" + "=" * 60)
    console.print("[bold blue]TRAINING ALL GRADIENT BOOSTING MODELS")
    console.print("[bold blue]" + "=" * 60)
    
    # Load data
    console.print("\n[yellow]Loading data...[/yellow]")
    X, y, feature_cols = load_event_level_data(data_path)
    
    console.print(f"  Samples: {len(X):,}")
    console.print(f"  Features: {len(feature_cols)}")
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
    
    results = {}
    models = {}
    
    # ========================================
    # 1. LightGBM
    # ========================================
    console.print("\n[bold]Training LightGBM...[/bold]")
    import lightgbm as lgb
    
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    lgb_train = lgb.Dataset(X_train.values, label=y_train.values)
    lgb_val = lgb.Dataset(X_val.values, label=y_val.values, reference=lgb_train)
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    
    start = time.time()
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_time = time.time() - start
    
    lgb_pred = lgb_model.predict(X_test.values)
    results["LightGBM"] = {
        "roc_auc": roc_auc_score(y_test, lgb_pred),
        "pr_auc": average_precision_score(y_test, lgb_pred),
        "f1": f1_score(y_test, (lgb_pred >= 0.5).astype(int)),
        "train_time": lgb_time,
    }
    models["lightgbm"] = lgb_model
    
    # Save LightGBM
    lgb_model.save_model(str(output_dir / "lightgbm.txt"))
    console.print(f"  ✓ LightGBM trained in {lgb_time:.2f}s")
    
    # ========================================
    # 2. XGBoost
    # ========================================
    console.print("\n[bold]Training XGBoost...[/bold]")
    import xgboost as xgb
    
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    dval = xgb.DMatrix(X_val.values, label=y_val.values)
    
    start = time.time()
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    xgb_time = time.time() - start
    
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_test.values))
    results["XGBoost"] = {
        "roc_auc": roc_auc_score(y_test, xgb_pred),
        "pr_auc": average_precision_score(y_test, xgb_pred),
        "f1": f1_score(y_test, (xgb_pred >= 0.5).astype(int)),
        "train_time": xgb_time,
    }
    models["xgboost"] = xgb_model
    
    # Save XGBoost
    xgb_model.save_model(str(output_dir / "xgboost.json"))
    console.print(f"  ✓ XGBoost trained in {xgb_time:.2f}s")
    
    # ========================================
    # 3. CatBoost
    # ========================================
    console.print("\n[bold]Training CatBoost...[/bold]")
    from catboost import CatBoostClassifier
    
    cat_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
        thread_count=-1,
    )
    
    start = time.time()
    cat_model.fit(
        X_train.values, y_train.values,
        eval_set=(X_val.values, y_val.values),
        verbose=False,
    )
    cat_time = time.time() - start
    
    cat_pred = cat_model.predict_proba(X_test.values)[:, 1]
    results["CatBoost"] = {
        "roc_auc": roc_auc_score(y_test, cat_pred),
        "pr_auc": average_precision_score(y_test, cat_pred),
        "f1": f1_score(y_test, (cat_pred >= 0.5).astype(int)),
        "train_time": cat_time,
    }
    models["catboost"] = cat_model
    
    # Save CatBoost
    cat_model.save_model(str(output_dir / "catboost.cbm"))
    console.print(f"  ✓ CatBoost trained in {cat_time:.2f}s")
    
    # ========================================
    # Results Summary
    # ========================================
    console.print("\n" + "=" * 60)
    console.print("[bold]RESULTS SUMMARY[/bold]")
    console.print("=" * 60)
    
    table = Table(title="Model Performance")
    table.add_column("Model", style="cyan")
    table.add_column("ROC-AUC", style="green")
    table.add_column("PR-AUC", style="green")
    table.add_column("F1", style="yellow")
    table.add_column("Train Time", style="magenta")
    
    for name, metrics in results.items():
        table.add_row(
            name,
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['pr_auc']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['train_time']:.2f}s",
        )
    
    console.print(table)
    
    # Save metadata
    metadata = {
        "feature_columns": feature_cols,
        "models": list(results.keys()),
        "results": results,
        "data_stats": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_rate": float(y.mean()),
        },
    }
    
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature columns for inference
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
    
    console.print(f"\n[green]✓ All models saved to: {output_dir}[/green]")
    console.print(f"\nSaved files:")
    for f in output_dir.glob("*"):
        size_kb = f.stat().st_size / 1024
        console.print(f"  {f.name}: {size_kb:.1f} KB")
    
    return results, models


def main():
    parser = argparse.ArgumentParser(description="Train all gradient boosting models")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/synthetic/synthetic_events.csv"),
        help="Path to training data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for models"
    )
    args = parser.parse_args()
    
    if not args.data.exists():
        console.print(f"[red]Data file not found: {args.data}[/red]")
        return
    
    train_all_models(args.data, args.output)


if __name__ == "__main__":
    main()
