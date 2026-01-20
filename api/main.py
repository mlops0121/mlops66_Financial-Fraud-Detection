@app.get("/")
def root():
    """Root endpoint returning service status."""
    return {"status": "running"}


@app.get("/health")
def health():
    """Health check endpoint used for monitoring."""
    return {"status": "ok"}


@app.post("/predict_test")
def predict_test(limit: int = 5):
    """Predict fraud probabilities on cached Kaggle test transactions.

    Args:
        limit: Number of predictions to return.

    Returns:
        Dictionary with count and a list of predictions.
    """
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
