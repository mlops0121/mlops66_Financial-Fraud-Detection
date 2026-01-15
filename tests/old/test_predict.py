
import numpy as np
import pytest
from src.models.tabnet_trainer import TabNetTrainer

# Mock Config
class MockConfig:
    """Mock configuration for testing."""
    CAT_EMB_DIM = 1
    N_D = 8
    N_A = 8
    N_STEPS = 3
    GAMMA = 1.3
    LAMBDA_SPARSE = 1e-5
    LEARNING_RATE = 2e-2
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.9
    MASK_TYPE = 'sparsemax'
    DEVICE = 'cpu'
    MAX_EPOCHS = 1  # 1 epoch for fast testing
    BATCH_SIZE = 32
    CHECKPOINT_DIR = 'checkpoints'
    RESUME_TRAINING = False
    CHECKPOINT_EVERY = 10
    PATIENCE = 5
    VIRTUAL_BATCH_SIZE = 16
    MODEL_PATH = 'test_model_output'

@pytest.fixture
def dummy_data():
    """Create dummy preprocessed data for testing."""
    # 10 samples, 4 features
    X = np.random.rand(10, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=(10,))
    
    return {
        'X_train': X,
        'y_train': y,
        'X_valid': X,
        'y_valid': y,
        'cat_idxs': [],
        'cat_dims': []
    }

def test_model_output_shape_and_type(dummy_data):
    config = MockConfig()
    trainer = TabNetTrainer(config, dummy_data, verbose=False)
    
    # Train briefly to ensure model is fitted
    trainer.train()
    
    # Test Data (Batch of 5)
    X_test = np.random.rand(5, 4).astype(np.float32)
    
    # 1. Test predict()
    preds = trainer.predict(X_test)
    
    print("\n--- Predict Output ---")
    print(f"Type: {type(preds)}")
    print(f"Shape: {preds.shape}")
    print(f"Values: {preds}")
    
    assert isinstance(preds, np.ndarray), "Prediction should be a numpy array"
    assert preds.shape == (5,), "Shape should be (n_samples,)"
    assert np.all(np.isin(preds, [0, 1])), "Predictions should be 0 or 1 for binary classification"

    # 2. Test predict_proba()
    probs = trainer.predict_proba(X_test)
    
    print("\n--- Predict Proba Output ---")
    print(f"Type: {type(probs)}")
    print(f"Shape: {probs.shape}")
    print(f"Values:\n{probs}")
    
    assert isinstance(probs, np.ndarray), "Probabilities should be a numpy array"
    assert probs.shape == (5, 2), "Shape should be (n_samples, n_classes)"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities should be between 0 and 1"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities for each sample should sum to 1"

if __name__ == "__main__":
    # Allow running this file directly to see print outputs
    from src.models.tabnet_trainer import TabNetTrainer
    d_data = dummy_data()
    test_model_output_shape_and_type(d_data)
