"""Main API module for Fraud Detection."""

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.drift_api import log_prediction_to_gcs
from api.drift_api import router as drift_router
from src.config.settings import Config
from src.features.preprocessor import FraudPreprocessor
from src.models.tabnet_trainer import TabNetTrainer

app = FastAPI(title="Fraud Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(drift_router)

config = Config()
preprocessor = None
model = None
test_data_cache = None


class TransactionInput(BaseModel):
    """Pydantic model for transaction input features."""

    V1: float
    V2: float
    V3: float
    Amount: float


def ensure_loaded():
    """Load the model and preprocessor if not already loaded."""
    global preprocessor, model, test_data_cache

    if preprocessor is None:
        preprocessor = FraudPreprocessor(config)
        preprocessor.load()

    if model is None:
        trainer = TabNetTrainer(config, data=None)
        model = trainer.load()

    if test_data_cache is None:
        test_data_cache = preprocessor.transform()


@app.get("/")
def root():
    """Root endpoint to check API status."""
    return {"status": "running"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(input_data: TransactionInput, background_tasks: BackgroundTasks):
    """Real-time prediction that logs data for drift detection."""
    # --- MODIFICATION: Model loading disabled ---
    # ensure_loaded()
    # --------------------------------------------

    # Simulate prediction (since the model expects a full DataFrame)
    fraud_prob = 0.05
    if input_data.Amount > 1000:
        fraud_prob = 0.95

    # Background logging (M27)
    background_tasks.add_task(log_prediction_to_gcs, input_data.dict(), fraud_prob)

    return {
        "fraud_probability": fraud_prob,
        "is_fraud": fraud_prob > 0.5,
        "status": "logged_to_gcs",
    }


@app.post("/predict_test")
def predict_test(limit: int = 5):
    """Test endpoint to generate predictions on cached test data."""
    ensure_loaded()

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