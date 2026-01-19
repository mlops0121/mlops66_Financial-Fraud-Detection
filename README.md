
# IEEE-CIS Fraud Detection with TabNet

A financial fraud detection system using TabNet deep learning model for the
[IEEE-CIS Fraud Detection Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection).

## ✨ Features

- 🧠 **TabNet Model** - Attention-based interpretable deep learning
- 📊 **Complete Preprocessing** - Auto handling of missing values, rare categories, feature encoding
- 🔄 **Checkpoint Support** - Resume training from interruption
- 📈 **Uncertainty Analysis** - Prediction confidence stratification
- 🎯 **Class Imbalance Handling** - Automatic class weight calculation
- 🌐 **FastAPI Inference API (M22)** - Run inference through a simple API endpoint

## 📁 Project Structure

```bash
├── api/                       # FastAPI application
│   ├── main.py                # Entry point for the API application
│   └── schemas.py             # Pydantic models for data validation
├── reports/                   # Reporting modules
│   └── report.py              # Script to generate performance reports
├── src/                       # Modular source code
│   ├── config/                # Configuration module (settings.py)
│   ├── data/                  # Data loading module (loader.py)
│   ├── evaluation/            # Evaluation module (metrics & uncertainty)
│   ├── features/              # Feature engineering (preprocessor, encoders, time_features)
│   ├── models/                # Model architecture & training (TabNet, callbacks)
│   └── utils/                 # Utility module (helpers.py)
├── tests/                     # Unit & Integration tests (Pytest)
├── data/                      # Dataset directory (Kaggle files go here)
├── checkpoints/               # Model checkpoints storage
├── Dockerfile                 # Docker configuration for containerization
├── docker-entrypoint.sh       # Entry script for Docker container
├── locustfile.py              # Load testing configuration (Locust)
├── train.py                   # Training entry point
├── predict.py                 # Prediction entry point (Kaggle submission)
├── preprocess.py              # Data preprocessing entry point
├── pyproject.toml             # Project configuration & dependencies
├── requirements.txt           # Production dependencies
├── ieee_cis_preprocessor.pkl  # Serialized preprocessor object
└── tabnet_fraud_model.zip     # Compressed model artifact
```
shell
Copy code

## 🚀 Quick Start

### 1) Environment Setup

```bash
pip install -r requirements.txt
2) Data Preparation
Download the Kaggle IEEE-CIS dataset and place these files in data/:

train_transaction.csv

train_identity.csv

test_transaction.csv

test_identity.csv

sample_submission.csv

3) Run
bash
Copy code
# Analyze data quality (optional)
python preprocess.py --analyze

# Preprocess data
python preprocess.py

# Train model
python train.py

# Predict (Kaggle submission)
python predict.py
🌐 FastAPI Inference API (M22)
Start the server
bash
Copy code
python -m uvicorn api.main:app --reload
Swagger docs:

http://127.0.0.1:8000/docs

Run inference (demo endpoint)
This endpoint runs preprocessing + TabNet inference on the Kaggle test set and returns the first limit predictions:

h
Copy code
POST /predict_test?limit=5
Example response includes:

TransactionID

fraud_probability

is_fraud
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_EPOCHS` | 100 | Maximum training epochs |
| `PATIENCE` | 10 | Early stopping patience |
| `BATCH_SIZE` | 8192 | Batch size |
| `CHECKPOINT_EVERY` | 10 | Checkpoint save interval |
| `RESUME_TRAINING` | True | Resume from checkpoint |

⚙️ Configuration
Modify parameters in src/config/settings.py.

📊 Model Performance
Test AUC: ~0.81

Top 5 Features: V230, P_emaildomain, M6, id_11, V154

📝 License
MIT License

