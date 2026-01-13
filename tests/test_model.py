import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.tabnet_trainer import TabNetTrainer

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
    DEVICE = 'cpu' if not torch.cuda.is_available() else 'cuda' # Adaptive device (should handle both Linux/Windows + macOS)
    MAX_EPOCHS = 5
    BATCH_SIZE = 32
    CHECKPOINT_DIR = 'checkpoints'


@pytest.fixture
def dummy_data():
    """Create dummy preprocessed data for testing."""
    X = np.random.rand(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=(100,))
    return {
        'X_train': X[:80],
        'y_train': y[:80],
        'X_valid': X[80:],
        'y_valid': y[80:],
        'cat_idxs': [],
        'cat_dims': []
    }

@pytest.fixture
def trainer(dummy_data):
    """Fixture for TabNetTrainer with mock config and dummy data."""
    config = MockConfig()
    return TabNetTrainer(config, dummy_data, verbose=False)

def test_init(trainer):
    assert trainer.model is None
    assert trainer.config.CAT_EMB_DIM == 1
    assert trainer.data['X_train'].shape == (80, 10)
    assert trainer.data['y_train'].shape == (80,)

def test_create_model(trainer):
    model = trainer._create_model()
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    assert model.device == trainer.config.DEVICE

