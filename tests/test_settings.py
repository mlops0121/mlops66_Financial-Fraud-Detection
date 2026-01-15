"""Tests for configuration settings."""

# Testing src/config/settings.py
import pytest
from src.config.settings import Config

@pytest.fixture
def config():
    """Create a Config instance."""
    return Config()

def test_config_attributes(config):
    """Test standard config attributes."""
    assert config.TRAIN_TRANSACTION == 'data/train_transaction.csv'
    assert config.TRAIN_IDENTITY == 'data/train_identity.csv'
    assert config.TEST_TRANSACTION == 'data/test_transaction.csv'
    assert config.TEST_IDENTITY == 'data/test_identity.csv'
    assert config.SAMPLE_SUBMISSION == 'data/sample_submission.csv'
    assert config.CHECKPOINT_DIR == 'checkpoints'
    assert config.MODEL_PATH == 'tabnet_fraud_model'
    assert config.PREPROCESSOR_PATH == 'ieee_cis_preprocessor.pkl'
    assert config.SUBMISSION_PATH == 'submission.csv'
    assert config.MISSING_THRESHOLD == 0.9
    assert config.RARE_CATEGORY_THRESHOLD == 100
    assert config.USE_TIME_SPLIT is True
    assert config.MAX_EPOCHS == 100
    assert config.RANDOM_SEED == 42
    assert config.BATCH_SIZE == 8192
    assert config.VIRTUAL_BATCH_SIZE == 256
    assert config.LEARNING_RATE == 5e-3
    assert config.N_D == 48
    assert config.N_A == 48
    assert config.N_STEPS == 4
    assert config.GAMMA == 1.5
    assert config.CAT_EMB_DIM == 2
    assert config.LAMBDA_SPARSE == 1e-3
    assert config.MASK_TYPE == 'entmax'
    assert config.PATIENCE == 10
    assert config.SCHEDULER_STEP_SIZE == 50
    assert config.SCHEDULER_GAMMA == 0.9
    assert config.DEVICE in ['cpu', 'cuda', 'mps']  # Depending on the environment
