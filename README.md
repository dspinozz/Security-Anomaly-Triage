# Security Event Anomaly Detection & Alert Triage Engine

[![Tests](https://img.shields.io/badge/tests-29%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

A cybersecurity ML system for anomaly scoring and alert triage using gradient boosted trees (LightGBM, XGBoost, CatBoost).

## 🎯 Project Goals

1. **Anomaly Detection**: Score security events for anomalousness
2. **Alert Triage**: Prioritize and classify alerts by severity
3. **Explainability**: Provide top contributing features for each score
4. **Production Ready**: FastAPI endpoints with JSON I/O

## 📊 Performance

Evaluated on UNSW-NB15-like dataset (50,000 samples, 9 attack types):

| Model | ROC-AUC | PR-AUC | F1 Score | Train Time |
|-------|---------|--------|----------|------------|
| LightGBM | 0.957 | 0.917 | 0.813 | 165s |
| XGBoost | 0.957 | 0.918 | 0.813 | 54s |
| CatBoost | 0.957 | 0.916 | 0.808 | 0.7s |
| Isolation Forest | 0.606 | 0.422 | 0.347 | 0.2s |

*Note: All three GBT variants achieve similar performance, which is expected—they use the same underlying algorithm family. See [notebooks/eda.py](notebooks/eda.py) for analysis.*

### Model Comparison Insights

- **Why similar scores?** All three are gradient boosted tree implementations
- **LightGBM**: Best for large datasets, memory efficient
- **XGBoost**: Most battle-tested, excellent documentation
- **CatBoost**: Fastest training, handles categoricals natively

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Event Stream                          │
│  (Network flows, auth logs, system events)              │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Engineering                         │
│  • Per-window aggregates (counts, entropy, ratios)      │
│  • Rolling statistics (z-scores, burstiness)            │
│  • Entity features (IP, user, port patterns)            │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    Scoring                              │
│  • Baseline rules (thresholds, known-bad patterns)      │
│  • LightGBM/XGBoost/CatBoost classifiers                │
│  • Isolation Forest (unsupervised)                      │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 Triage Output                           │
│  • Anomaly score (0-1)                                  │
│  • Predicted class (NORMAL/ATTACK)                      │
│  • Top-K contributing features                          │
│  • Severity ranking (critical/high/medium/low/info)     │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
sec-anomaly-triage/
├── README.md
├── requirements.txt
├── configs/
│   ├── features.yaml         # Feature definitions
│   └── thresholds.yaml       # Baseline rule thresholds
├── data/
│   └── unsw-nb15/            # UNSW-NB15 dataset
├── src/
│   ├── features/
│   │   ├── engineering.py    # Feature engineering pipeline
│   │   └── windows.py        # Time window aggregations
│   ├── models/
│   │   ├── baseline.py       # Rule-based scorer
│   │   ├── lightgbm_model.py # LightGBM classifier
│   │   └── isolation.py      # Isolation Forest
│   ├── scoring/
│   │   ├── scorer.py         # Unified scoring interface
│   │   └── explainer.py      # Feature importance
│   └── utils/
│       └── metrics.py        # Evaluation metrics
├── api/
│   └── main.py               # FastAPI endpoints
├── notebooks/
│   └── eda.py                # Exploratory data analysis
├── scripts/
│   ├── download_data.py      # Synthetic data generator
│   ├── download_unsw.py      # UNSW-NB15 preparation
│   ├── train.py              # Training script
│   ├── train_all.py          # Multi-model training
│   └── compare_models.py     # Model comparison benchmark
├── tests/
│   ├── test_api.py           # API integration tests
│   ├── test_features.py      # Feature engineering tests
│   └── test_models.py        # Model tests
└── models/
    └── trained/              # Saved model artifacts
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/dspinozz/Security-Anomaly-Triage.git
cd sec-anomaly-triage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Prepare Data

```bash
# Download UNSW-NB15-like dataset (recommended)
python scripts/download_unsw.py --sample 50000

```

### Train Models

```bash
# Train all models (LightGBM, XGBoost, CatBoost, Isolation Forest)
python scripts/train_all.py --data data/unsw-nb15/unsw_nb15_demo_50000.csv

# Run model comparison benchmark
python scripts/compare_models.py
```

### Run Tests

```bash
pytest tests/ -v
```

### Start API Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

## 🔌 API Endpoints

### Health Check
```bash
curl http://localhost:8001/health
```

### Score Single Event
```bash
curl -X POST http://localhost:8001/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "dst_port": 22,
    "protocol": "ssh",
    "status": "failed",
    "bytes_in": 100,
    "bytes_out": 50
  }'
```

Response:
```json
{
  "anomaly_score": 0.9942,
  "classification": "ATTACK",
  "severity": "critical",
  "model_scores": {
    "lightgbm": 0.983,
    "xgboost": 0.9997,
    "catboost": 1.0
  },
  "features_used": {
    "dst_port": 22.0,
    "proto_ssh": 1.0,
    "status_failed": 1.0
  }
}
```

### Batch Scoring
```bash
curl -X POST http://localhost:8001/v1/score/batch \
  -H "Content-Type: application/json" \
  -d '{
    "events": [
      {"dst_port": 443, "protocol": "https"},
      {"dst_port": 22, "protocol": "ssh", "status": "failed"}
    ]
  }'
```

## 📊 Dataset

This project supports multiple datasets:

| Dataset | Type | Size | Use Case |
|---------|------|------|----------|
| Synthetic | Generated | 50K | Quick demos, CI/CD |
| UNSW-NB15 | Real | 2.5M | Production training |
| CICIDS2017 | Real | 2.8M | Alternative real data |

### Attack Types (UNSW-NB15)

- **Normal**: Benign network traffic
- **DoS**: Denial of Service attacks
- **Exploits**: Vulnerability exploitation
- **Fuzzers**: Input fuzzing attacks
- **Generic**: Generic attack patterns
- **Reconnaissance**: Network scanning
- **Backdoor**: Backdoor malware
- **Analysis**: Malware analysis evasion
- **Shellcode**: Shell injection
- **Worms**: Self-propagating malware

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

Test coverage:
- ✅ API endpoints (health, scoring, batch)
- ✅ Feature engineering (windows, aggregation)
- ✅ Model training and inference
- ✅ Model persistence (save/load)

## 📈 EDA Insights

Run exploratory data analysis:
```bash
python notebooks/eda.py
```

Key findings:
- Class imbalance: ~2:1 (Normal:Attack)
- Top predictive features: `sloss`, `dloss`, `spkts`, `rate`
- DoS attacks: High rate (1000+ vs 100 normal), short duration
- Reconnaissance: Low bytes, many connections

## 🔧 Configuration

### Feature Configuration (`configs/features.yaml`)
```yaml
windows:
  - name: 1min
    duration_seconds: 60
  - name: 5min
    duration_seconds: 300
  - name: 15min
    duration_seconds: 900
```

### Threshold Configuration (`configs/thresholds.yaml`)
```yaml
rules:
  - name: high_event_volume
    severity: medium
    conditions:
      - field: event_count_1min
        operator: ">"
        value: 100
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

Built for cybersecurity ML portfolio demonstration. Emphasizes:
- Classic ML (tabular data, gradient boosted trees)
- Production engineering (API, tests, configuration)
- Explainability and interpretability
