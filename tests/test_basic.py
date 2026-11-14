"""Basic sanity test to verify test harness and package imports."""

import paragai
from paragai.config import ModelConfig, TrainConfig
from paragai.utils.seed import set_seed


def test_package_metadata():
    assert paragai.__version__ == "0.1.0"


def test_config_initialization():
    config = ModelConfig()
    assert config.n_embd == 192
    assert config.n_head == 6
    assert config.head_dim == 32

    train_config = TrainConfig()
    assert train_config.batch_size == 32
    assert train_config.learning_rate == 6e-4


def test_seed_determinism():
    set_seed(42)
