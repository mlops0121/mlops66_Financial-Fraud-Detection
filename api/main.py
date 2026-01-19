"""FastAPI application for fraud detection predictions."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import Config
from src.features.preprocessor import FraudPreprocessor
from src.models.tabnet_trainer import TabNetTrainer

app = FastAPI(title="Fraud Detection API", version="1.0")

# Enable CORS (so browser frontend can call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects (loaded once)
config = Config()
preprocessor = None
model = None
test_data_cache = None  # ✅ cached preprocessed test data


@app.on_event("startup")
def load_all():
    """Load model + preprocessor once when the server starts, and cache test preprocessing."""
    global preprocessor, model, test_data_cache

    # Load preprocessor
    preprocessor = FraudPreprocessor(config)
    preprocessor.load()

    # Load model
    trainer = TabNetTrainer(config, data=None)
    model = trainer.load()

    # Cache transformed test set (so API is fast)
    test_data_cache = preprocessor.transform()


@app.get("/")
def root():
    """Return the API status."""
    return {"status": "running"}


@app.get("/health")
def health():
    """Health check endpoint (useful for deployment/monitoring)."""
    return {"status": "ok"}


@app.post("/predict_test")
def predict_test(limit: int = 5):
    """Return fraud predictions for the first N cached test transactions."""
    proba = model.predict_proba(test_data_cache["X_test"])[:, 1]

    out = []
    for tid, p in zip(test_data_cache["transaction_ids"][:limit], proba[:limit]):
        out.append(
            {
                "TransactionID": int(tid),
                "fraud_probability": float(p),
                "is_fraud": bool(p >= 0.5),
            }
        )

    return {"count": limit, "predictions": out}
