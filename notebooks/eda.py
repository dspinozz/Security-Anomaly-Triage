#!/usr/bin/env python3
"""
Security Event Anomaly Detection - Exploratory Data Analysis

Run this script to generate EDA visualizations and insights.
This can be converted to a Jupyter notebook with: jupytext --to notebook eda.py

Usage:
    python notebooks/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 60)
print("SECURITY ANOMALY DETECTION - EDA")
print("=" * 60)

# Load dataset
data_path = Path("data/unsw-nb15/unsw_nb15_demo_50000.csv")
if not data_path.exists():
    data_path = Path("../data/unsw-nb15/unsw_nb15_demo_50000.csv")
    
df = pd.read_csv(data_path)

print(f"\n📊 DATASET OVERVIEW")
print(f"  Shape: {df.shape}")
print(f"  Total Samples: {len(df):,}")
print(f"  Features: {len(df.columns)}")

# Class distribution
print(f"\n⚖️ CLASS DISTRIBUTION")
label_counts = df['label'].value_counts()
print(f"  Normal (0): {label_counts[0]:,} ({100*label_counts[0]/len(df):.1f}%)")
print(f"  Attack (1): {label_counts[1]:,} ({100*label_counts[1]/len(df):.1f}%)")
print(f"  Imbalance Ratio: 1:{label_counts[0]/label_counts[1]:.1f}")

# Attack category distribution
print(f"\n🎯 ATTACK CATEGORIES")
attack_cats = df['attack_cat'].value_counts()
for cat, count in attack_cats.items():
    pct = 100 * count / len(df)
    bar = "█" * int(pct / 2)
    print(f"  {cat:15} {count:6,} ({pct:5.1f}%) {bar}")

# Feature statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['label']]

print(f"\n📈 FEATURE STATISTICS")
print(f"  Numeric features: {len(numeric_cols)}")
print(f"  Categorical features: {len(df.columns) - len(numeric_cols) - 2}")

# Missing values
missing = df.isnull().sum().sum()
print(f"  Missing values: {missing}")

# Correlation with target
print(f"\n🔗 TOP FEATURES CORRELATED WITH ATTACK LABEL")
correlations = df[numeric_cols + ['label']].corr()['label'].drop('label')
correlations = correlations.sort_values(key=abs, ascending=False)

for feature in correlations.head(10).index:
    corr = correlations[feature]
    direction = "↑" if corr > 0 else "↓"
    print(f"  {feature:25} {direction} {corr:+.4f}")

# Attack signatures
print(f"\n🔍 ATTACK SIGNATURES (Mean Values)")
print(f"  {'Attack Type':15} {'Duration':>10} {'Src Bytes':>12} {'Rate':>10}")
print("  " + "-" * 50)

for attack in df['attack_cat'].unique():
    subset = df[df['attack_cat'] == attack]
    dur = subset['dur'].mean()
    sbytes = subset['sbytes'].mean()
    rate = subset['rate'].mean()
    print(f"  {attack:15} {dur:10.3f} {sbytes:12.0f} {rate:10.1f}")

# Key findings
print(f"\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)
print("""
1. CLASS BALANCE
   - Dataset is moderately imbalanced (~2:1 ratio)
   - Manageable with class weights or SMOTE

2. FEATURE INSIGHTS
   - Duration (dur): Attacks tend to be shorter
   - Byte counts: Anomalies in data transfer patterns
   - Rate: DoS attacks show significantly higher rates
   
3. ATTACK PATTERNS
   - DoS: High rate, short duration
   - Reconnaissance: Many connections, low bytes
   - Exploits: Unusual payload sizes
   
4. MODELING RECOMMENDATIONS
   - Tree-based models should work well
   - Consider feature engineering on duration/bytes
   - Class weighting recommended
""")

print("\n✓ EDA Complete!")
