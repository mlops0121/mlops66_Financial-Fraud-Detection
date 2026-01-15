import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch # For mocking dependencies

from src.models.tabnet_trainer import TabNetTrainer
from src.utils.helpers import find_latest_checkpoint
from src.data.loader import DataLoader
from src.features.preprocessor import FraudPreprocessor
from src.config.settings import Config

class MockConfig:
    """Mock configuration for testing."""
    def get_best_device(self):
        if torch.cuda.is_available(): # Check for CUDA availability
            return 'cuda'
        elif torch.backends.mps.is_available(): # Check for MPS availability (macOS [M1/M2/M3])
            return 'mps'
        else:
            return 'cpu' # Fallback to CPU
        
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
    DEVICE = get_best_device() # Adaptive device (should handle both Linux/Windows + macOS)
    MAX_EPOCHS = 5
    BATCH_SIZE = 32
    CHECKPOINT_DIR = 'checkpoints'





