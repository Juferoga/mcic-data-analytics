"""
SOM Analyzer - Quality metrics for Self-Organizing Maps.

This module provides the SOMAnalyzer class for computing
quality metrics and analyzing SOM performance.

Quality Metrics Explained
=========================

Quantization Error (QE)
------------------------
Average distance from each sample to its BMU.

    QE = (1/N) * Σ ||x_i - w_BMU(x_i)||

Lower is better, but:
- Very low QE may indicate overfitting
- Too high QE means poor representation

Typical values (after normalization to [0,1]):
- Good: 0.05 - 0.15
- Acceptable: 0.15 - 0.3
- Poor: > 0.3

Topographic Error (TE)
-----------------------
Proportion of samples whose first and second BMUs
are not adjacent in the grid.

    TE = (1/N) * Σ adjacent(first_BMU, second_BMU) ? 0 : 1

Where adjacent() returns True if BMUs are neighbors.

Lower is better:
- Good TE: < 0.1 (10% of samples have non-adjacent BMUs)
- Acceptable TE: < 0.2
- Poor TE: > 0.2 (topology not preserved well)

Hit Distribution
----------------
How samples are distributed across neurons.
- Even distribution: All neurons have similar hits
- Uneven distribution: Some neurons over/under-used

Ideal: Poisson-like distribution
Problem: Many empty neurons (wasted capacity)
Problem: Very high hits on few neurons (overcrowding)
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .trainer import SOMTrainer
from .exceptions import NotTrainedError


class SOMAnalyzer:
    """
    Analyze SOM quality and characteristics.

    Provides methods to compute quality metrics and
    understand the SOM's representation of the data.

    Example:
        >>> trainer = SOMTrainer(x=10, y=10, input_len=4)
        >>> trainer.fit(data)
        >>>
        >>> analyzer = SOMAnalyzer(trainer)
        >>>
        >>> # Get quality metrics
        >>> metrics = analyzer.get_metrics()
        >>> print(f"Quantization Error: {metrics['qe']:.4f}")
        Quantization Error: 0.1234
        >>>
        >>> # Get node distribution
        >>> distribution = analyzer.node_distribution()
        >>> print(f"Nodes used: {(distribution > 0).sum()}")
        Nodes used: 67
    """

    def __init__(self, trainer: SOMTrainer):
        """
        Initialize the analyzer.

        Args:
            trainer: A trained SOMTrainer instance.

        Raises:
            NotTrainedError: If the trainer hasn't been trained.
        """
        if not trainer.is_trained:
            raise NotTrainedError("Cannot create analyzer: SOM is not trained.")

        self._trainer = trainer
        self._som = trainer._som
        self._config = trainer.config

    def quantization_error(self, data: np.ndarray) -> float:
        """
        Calculate the average quantization error.

        Args:
            data: Input data array (n_samples, n_features).

        Returns:
            Float: Average quantization error across all samples.

        Interpretation:
            - < 0.1: Excellent representation
            - 0.1 - 0.2: Good representation
            - 0.2 - 0.3: Acceptable
            - > 0.3: Poor representation (consider larger grid)
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Normalize using training parameters
        data_range = self._trainer._data_max - self._trainer._data_min
        data_range[data_range == 0] = 1
        normalized_data = (data - self._trainer._data_min) / data_range

        return float(self._som.quantization_error(normalized_data))

    def topographic_error(self, data: np.ndarray, n_samples: int = 1000) -> float:
        """
        Calculate the topographic error.

        The topographic error measures how well the topological
        structure of the data is preserved in the SOM.

        Args:
            data: Input data array.
            n_samples: Number of samples to use (for efficiency).
                If data is smaller, uses all samples.

        Returns:
            Float: Proportion of samples with non-adjacent first
            and second BMUs. Lower is better.

        Interpretation:
            - < 0.1: Excellent topology preservation
            - 0.1 - 0.2: Good topology
            - > 0.2: Poor topology (training may need more epochs)
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Normalize using training parameters
        data_range = self._trainer._data_max - self._trainer._data_min
        data_range[data_range == 0] = 1
        normalized_data = (data - self._trainer._data_min) / data_range

        return float(self._som.topographic_error(normalized_data[:n_samples]))

    def node_distribution(self, data: np.ndarray) -> np.ndarray:
        """
        Get the distribution of samples across nodes.

        Returns a 2D array where each cell contains the number
        of samples assigned to that neuron.

        Args:
            data: Input data array.

        Returns:
            2D numpy array of shape (y, x) with hit counts.

        Example:
            >>> hits = analyzer.node_distribution(data)
            >>> print(f"Max hits on single node: {hits.max()}")
            Max hits on single node: 15
            >>> print(f"Empty nodes: {(hits == 0).sum()}")
            Empty nodes: 12
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        assignments = self._trainer.transform(data)

        hits = np.zeros((self._config.y, self._config.x))
        for _, row in assignments.iterrows():
            hits[int(row["neuron_y"]), int(row["neuron_x"])] += 1

        return hits

    def get_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Get all quality metrics at once.

        Args:
            data: Input data for evaluation.

        Returns:
            Dictionary with metrics:
            - qe: Quantization error
            - te: Topographic error
            - nodes_used: Number of neurons with at least 1 sample
            - total_nodes: Total number of neurons
            - coverage: Proportion of neurons used
            - max_hits: Maximum samples on single node
            - mean_hits: Average samples per used node

        Example:
            >>> metrics = analyzer.get_metrics(data)
            >>> for key, value in metrics.items():
            ...     print(f"{key}: {value}")
            qe: 0.1234
            te: 0.0567
            nodes_used: 67
            total_nodes: 100
            coverage: 0.67
            max_hits: 15
            mean_hits: 3.2
        """
        hits = self.node_distribution(data)

        metrics = {
            "qe": self.quantization_error(data),
            "te": self.topographic_error(data),
            "nodes_used": int((hits > 0).sum()),
            "total_nodes": self._config.x * self._config.y,
            "coverage": float((hits > 0).sum()) / (self._config.x * self._config.y),
            "max_hits": int(hits.max()),
            "mean_hits": float(hits[hits > 0].mean()) if (hits > 0).any() else 0.0,
        }

        return metrics
