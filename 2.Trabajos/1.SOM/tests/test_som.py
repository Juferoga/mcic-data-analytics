"""Tests for SOM module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from analitica.som import (
    SOMTrainer,
    SOMPredictor,
    SOMVisualizer,
    SOMAnalyzer,
    SOMConfig,
    SOMError,
    NotTrainedError,
    InvalidConfigurationError,
    InsufficientDataError,
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    data = np.random.rand(100, 4)  # 100 samples, 4 features
    return data


@pytest.fixture
def trained_trainer(sample_data):
    """Create and train a SOM trainer."""
    trainer = SOMTrainer(x=5, y=5, input_len=4, random_seed=42)
    trainer.fit(sample_data, epochs=10)
    return trainer


class TestSOMTrainer:
    """Test SOMTrainer class."""

    def test_initialization(self):
        """Test SOMTrainer initialization."""
        trainer = SOMTrainer(x=10, y=10, input_len=4)
        assert trainer.config.x == 10
        assert trainer.config.y == 10
        assert trainer.config.input_len == 4

    def test_invalid_grid_size(self):
        """Test error on invalid grid size."""
        with pytest.raises(InvalidConfigurationError):
            SOMTrainer(x=0, y=10)

    def test_fit_numpy(self, sample_data):
        """Test fitting with numpy array."""
        trainer = SOMTrainer(x=5, y=5, input_len=4)
        trainer.fit(sample_data)
        assert trainer.is_trained

    def test_fit_dataframe(self, sample_data):
        """Test fitting with DataFrame."""
        df = pd.DataFrame(sample_data, columns=["a", "b", "c", "d"])
        trainer = SOMTrainer(x=5, y=5, input_len=4)
        trainer.fit(df)
        assert trainer.is_trained

    def test_transform(self, trained_trainer, sample_data):
        """Test transform returns neuron assignments."""
        result = trained_trainer.transform(sample_data)
        assert "neuron_x" in result.columns
        assert "neuron_y" in result.columns
        assert len(result) == len(sample_data)

    def test_fit_transform(self, sample_data):
        """Test fit_transform convenience method."""
        result = SOMTrainer(x=5, y=5, input_len=4).fit_transform(sample_data, epochs=10)
        assert "neuron_x" in result.columns
        assert "neuron_y" in result.columns

    def test_insufficient_data(self):
        """Test error on insufficient data."""
        data = np.random.rand(2, 4)  # Only 2 samples
        trainer = SOMTrainer(x=5, y=5, input_len=4)
        with pytest.raises(InsufficientDataError):
            trainer.fit(data)


class TestSOMPredictor:
    """Test SOMPredictor class."""

    def test_bmu(self, trained_trainer):
        """Test BMU finding."""
        predictor = SOMPredictor(trained_trainer)
        sample = trained_trainer._data_min + np.random.rand(4) * 0.1
        bmu = predictor.bmu(sample)
        assert isinstance(bmu, tuple)
        assert 0 <= bmu[0] < trained_trainer.config.x
        assert 0 <= bmu[1] < trained_trainer.config.y

    def test_quantization_error(self, trained_trainer):
        """Test quantization error calculation."""
        predictor = SOMPredictor(trained_trainer)
        sample = np.random.rand(4)
        error = predictor.quantization_error(sample)
        assert isinstance(error, float)
        assert error >= 0

    def test_not_trained_error(self):
        """Test error when trainer not trained."""
        trainer = SOMTrainer(x=5, y=5, input_len=4)
        with pytest.raises(NotTrainedError):
            SOMPredictor(trainer)


class TestSOMAnalyzer:
    """Test SOMAnalyzer class."""

    def test_quantization_error(self, trained_trainer, sample_data):
        """Test QE calculation."""
        analyzer = SOMAnalyzer(trained_trainer)
        qe = analyzer.quantization_error(sample_data)
        assert isinstance(qe, float)
        assert qe >= 0

    def test_topographic_error(self, trained_trainer, sample_data):
        """Test TE calculation."""
        analyzer = SOMAnalyzer(trained_trainer)
        te = analyzer.topographic_error(sample_data)
        assert isinstance(te, float)
        assert 0 <= te <= 1

    def test_node_distribution(self, trained_trainer, sample_data):
        """Test node distribution."""
        analyzer = SOMAnalyzer(trained_trainer)
        hits = analyzer.node_distribution(sample_data)
        assert hits.shape == (5, 5)
        assert hits.sum() == len(sample_data)

    def test_get_metrics(self, trained_trainer, sample_data):
        """Test get_metrics returns all metrics."""
        analyzer = SOMAnalyzer(trained_trainer)
        metrics = analyzer.get_metrics(sample_data)
        assert "qe" in metrics
        assert "te" in metrics
        assert "nodes_used" in metrics
        assert "coverage" in metrics


class TestSOMVisualizer:
    """Test SOMVisualizer class."""

    def test_plot_umatrix(self, trained_trainer):
        """Test U-Matrix plot creation."""
        import matplotlib.pyplot as plt

        visualizer = SOMVisualizer(trained_trainer)
        fig = visualizer.plot_umatrix(show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_component_planes(self, trained_trainer):
        """Test component planes plot."""
        import matplotlib.pyplot as plt

        visualizer = SOMVisualizer(trained_trainer)
        fig = visualizer.plot_component_planes(show=False)
        assert fig is not None
        plt.close(fig)

    def test_plot_bmu(self, trained_trainer, sample_data):
        """Test BMU highlighting."""
        import matplotlib.pyplot as plt

        visualizer = SOMVisualizer(trained_trainer)
        fig = visualizer.plot_bmu(sample_data[0], show=False)
        assert fig is not None
        plt.close(fig)


class TestSOMConfig:
    """Test SOMConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = SOMConfig(x=10, y=10, input_len=4)
        assert config.x == 10

    def test_invalid_x(self):
        """Test invalid x dimension."""
        with pytest.raises(InvalidConfigurationError):
            SOMConfig(x=-1, y=10)

    def test_invalid_learning_rate(self):
        """Test invalid learning rate."""
        with pytest.raises(InvalidConfigurationError):
            SOMConfig(x=10, y=10, learning_rate=1.5)
