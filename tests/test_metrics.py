import pytest
import numpy as np
from src.evaluation.metrics import Metrics
from pytorch_tabnet.tab_model import TabNetClassifier
from src.models.tabnet_trainer import TabNetTrainer
from sklearn.datasets import make_classification


@pytest.fixture
def metrics():
    return Metrics()

@pytest.fixture
def dummy_data():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    return X, y

def test_evaluate_model(model, dummy_data):
    X, y = dummy_data
    model = TabNetClassifier()
    model.fit(X, y, max_epochs=1, patience=1, batch_size=16, virtual_batch_size=8, num_workers=0, drop_last=False)
    metrics.evaluate_model(model, X, y)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.auc <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0

def test_feature_importance(metrics, dummy_data):
    X, y = dummy_data
    model = TabNetClassifier()
    model.fit(X, y, max_epochs=1, patience=1, batch_size=16, virtual_batch_size=8, num_workers=0, drop_last=False)
    feature_importance = metrics.get_feature_importance(model, [f"feature_{i}" for i in range(X.shape[1])])
    assert feature_importance is not None
    assert len(feature_importance) == X.shape[1]
    assert np.all(feature_importance >= 0)
    