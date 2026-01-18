#!/usr/bin/env python3
"""Hyperparameter tuning with Optuna.

This script demonstrates hyperparameter optimization for gradient boosted trees.
Unlike deep learning (SGD on model parameters), GBTs tune tree structure hyperparameters.

Usage:
    python scripts/tune_hyperparams.py --model lightgbm --trials 50
    python scripts/tune_hyperparams.py --model xgboost --trials 50
    python scripts/tune_hyperparams.py --model catboost --trials 50

Key Difference from Deep Learning:
    Deep Learning: Optimizes PARAMETERS (weights) via gradient descent
    GBTs: Optimizes HYPERPARAMETERS (tree structure) via black-box search
    
    Deep Learning optimizers (Adam, SGD) update millions of weights.
    Optuna searches discrete/continuous hyperparameter spaces.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler
import time
import warnings

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent


def load_data():
    """Load UNSW-NB15 dataset."""
    data_path = PROJECT_ROOT / "data" / "unsw-nb15" / "unsw_nb15_demo_50000.csv"
    
    if not data_path.exists():
        # Fall back to synthetic
        data_path = PROJECT_ROOT / "data" / "synthetic" / "synthetic_events.csv"
    
    df = pd.read_csv(data_path)
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'label']
    
    # Add categorical encoding for UNSW data
    for cat_col in ['proto', 'service', 'state']:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
            df = pd.concat([df, dummies], axis=1)
            numeric_cols.extend(dummies.columns.tolist())
    
    X = df[numeric_cols].fillna(0)
    y = df['label']
    
    return X, y


# ============================================================
# LIGHTGBM OBJECTIVE
# ============================================================
def lightgbm_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM.
    
    Hyperparameters tuned (NOT model parameters):
    - num_leaves: Number of leaves per tree (structure)
    - max_depth: Maximum tree depth (structure)
    - learning_rate: Shrinkage factor (not gradient step size!)
    - subsample: Row sampling ratio
    - colsample_bytree: Column sampling ratio
    - reg_alpha: L1 regularization
    - reg_lambda: L2 regularization
    - min_child_samples: Minimum samples per leaf
    """
    import lightgbm as lgb
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,
        
        # Hyperparameters to tune
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    
    preds = model.predict(X_val)
    return roc_auc_score(y_val, preds)


# ============================================================
# XGBOOST OBJECTIVE
# ============================================================
def xgboost_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for XGBoost."""
    import xgboost as xgb
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": 42,
        
        # Hyperparameters to tune
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    
    preds = model.predict(dval)
    return roc_auc_score(y_val, preds)


# ============================================================
# CATBOOST OBJECTIVE
# ============================================================
def catboost_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for CatBoost."""
    from catboost import CatBoostClassifier
    
    params = {
        "iterations": 500,
        "verbose": False,
        "random_seed": 42,
        "early_stopping_rounds": 50,
        
        # Hyperparameters to tune
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


def run_optimization(model_name: str, n_trials: int = 50):
    """Run Optuna optimization for specified model."""
    
    print("=" * 60)
    print(f"HYPERPARAMETER TUNING: {model_name.upper()}")
    print("=" * 60)
    
    # Load and split data
    print("\nLoading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Select objective function
    if model_name == "lightgbm":
        objective = lambda trial: lightgbm_objective(trial, X_train, y_train, X_val, y_val)
    elif model_name == "xgboost":
        objective = lambda trial: xgboost_objective(trial, X_train, y_train, X_val, y_val)
    elif model_name == "catboost":
        objective = lambda trial: catboost_objective(trial, X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=f"{model_name}_tuning",
    )
    
    # Optimize
    print(f"\nRunning {n_trials} trials...")
    start = time.time()
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,  # Sequential for reproducibility
    )
    
    elapsed = time.time() - start
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nBest trial:")
    print(f"  ROC-AUC: {study.best_value:.4f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/n_trials:.1f}s per trial)")
    
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    output_dir = PROJECT_ROOT / "models" / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": model_name,
        "best_roc_auc": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "elapsed_seconds": elapsed,
    }
    
    with open(output_dir / f"{model_name}_best_params.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to: {output_dir / f'{model_name}_best_params.json'}")
    
    # Compare to baseline
    print("\n" + "=" * 60)
    print("COMPARISON TO BASELINE")
    print("=" * 60)
    
    baseline_file = PROJECT_ROOT / "models" / "trained" / "model_metadata.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)
        
        baseline_roc = baseline["results"].get(
            {"lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}[model_name],
            {}
        ).get("roc_auc", 0)
        
        improvement = (study.best_value - baseline_roc) * 100
        print(f"  Baseline ROC-AUC: {baseline_roc:.4f}")
        print(f"  Tuned ROC-AUC:    {study.best_value:.4f}")
        print(f"  Improvement:      {improvement:+.2f}%")
    
    return study


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/tune_hyperparams.py --model lightgbm --trials 50
    python scripts/tune_hyperparams.py --model xgboost --trials 30
    python scripts/tune_hyperparams.py --model catboost --trials 20

Note: This tunes HYPERPARAMETERS (tree structure), not model PARAMETERS.
Deep learning uses SGD/Adam to optimize millions of weight parameters.
GBTs don't have learnable parameters in the same sense - trees are built
greedily, and we tune the structure constraints (depth, leaves, etc.).
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lightgbm", "xgboost", "catboost"],
        default="lightgbm",
        help="Model to tune"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials"
    )
    args = parser.parse_args()
    
    run_optimization(args.model, args.trials)


if __name__ == "__main__":
    main()
