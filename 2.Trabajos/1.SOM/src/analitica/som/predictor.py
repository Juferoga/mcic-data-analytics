"""
SOM Predictor - Make predictions with a trained SOM.

This module provides the SOMPredictor class for making predictions
with a trained Self-Organizing Map.

Best Matching Unit (BMU)
========================

The BMU is the neuron whose weight vector is most similar to the
input vector. Finding the BMU is fundamental to SOM operation.

For a given input vector x, the BMU b is found as:

    b = argmin_i ||x - w_i||

Where:
- w_i: Weight vector of neuron i
- ||...||: Euclidean distance
- argmin_i: Neuron i that minimizes the distance

Quantization Error
==================

The quantization error measures how well the SOM represents the data.
It's the average distance from each sample to its BMU:

    QE = (1/N) * Σ ||x_i - w_BMU(x_i)||

Lower QE = better representation.

Interpretation:
- Very low QE (< 0.05): Possible overfitting
- Low QE (0.05-0.15): Good representation
- Moderate QE (0.15-0.3): Acceptable
- High QE (> 0.3): Poor representation, consider larger grid
"""

from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd

from .trainer import SOMTrainer
from .exceptions import NotTrainedError


class SOMPredictor:
    """
    Make predictions with a trained SOM.

    This class wraps a trained SOMTrainer to provide prediction
    methods like BMU finding and distance calculations.

    Example:
        >>> trainer = SOMTrainer(x=10, y=10, input_len=4)
        >>> trainer.fit(data)
        >>>
        >>> predictor = SOMPredictor(trainer)
        >>>
        >>> # Find BMU for a single sample
        >>> sample = data[0]
        >>> bmu = predictor.bmu(sample)
        >>> print(f"BMU: ({bmu[0]}, {bmu[1]})")
        BMU: (3, 7)
        >>>
        >>> # Calculate distance from sample to its BMU
        >>> distance = predictor.quantization_error(sample)
        >>> print(f"Distance: {distance:.4f}")
        Distance: 0.1234
    """

    def __init__(self, trainer: SOMTrainer):
        """
        Initialize the predictor with a trained SOM.

        Args:
            trainer: A trained SOMTrainer instance.

        Raises:
            NotTrainedError: If the trainer hasn't been trained.
        """
        if not trainer.is_trained:
            raise NotTrainedError("Cannot create predictor: SOM is not trained.")

        self._trainer = trainer
        self._som = trainer._som
        self._config = trainer.config

    def bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Find the Best Matching Unit for a sample.

        The BMU is the neuron whose weight vector is most similar
        to the input vector x (using Euclidean distance).

        Args:
            x: Input vector as 1D array.
               Must have same length as input_len.

        Returns:
            Tuple of (x, y) coordinates of the BMU.
            x is column (0 to width-1)
            y is row (0 to height-1)

        Example:
            >>> sample = np.array([0.5, 0.3, 0.8, 0.2])
            >>> bmu = predictor.bmu(sample)
            >>> print(f"BMU coordinates: {bmu}")
            BMU coordinates: (3, 7)
        """
        # Normalize input using training parameters
        x = np.asarray(x).flatten()
        x = self._normalize(x)

        # Find winner
        winner = self._som.winner(x)
        return tuple(winner)

    def quantization_error(self, x: np.ndarray) -> float:
        """
        Calculate the quantization error for a sample.

        The quantization error is the Euclidean distance between
        the sample and its BMU's weight vector.

        Args:
            x: Input vector.

        Returns:
            Float: The quantization error (distance to BMU).

        Interpretation:
            - Low error: Sample is well-represented by its neuron
            - High error: Sample is far from any neuron (potential outlier)

        Example:
            >>> error = predictor.quantization_error(sample)
            >>> print(f"Quantization error: {error:.4f}")
            Quantization error: 0.1234
        """
        x = np.asarray(x).flatten()
        x_normalized = self._normalize(x)

        # Get BMU
        bmu = self._som.winner(x_normalized)

        # Get BMU weights
        weights = self._som.get_weights()
        bmu_weights = weights[bmu[1], bmu[0], :]

        # Calculate distance
        error = np.linalg.norm(x_normalized - bmu_weights)
        return float(error)

    def predict(self, x: np.ndarray) -> Tuple[int, int, float]:
        """
        Get complete prediction for a sample.

        Returns BMU coordinates and quantization error.

        Args:
            x: Input vector.

        Returns:
            Tuple of (neuron_x, neuron_y, distance).

        Example:
            >>> neuron_x, neuron_y, distance = predictor.predict(sample)
            >>> print(f"Node ({neuron_x}, {neuron_y}), distance: {distance:.4f}")
            Node (3, 7), distance: 0.1234
        """
        bmu = self.bmu(x)
        distance = self.quantization_error(x)
        return (bmu[0], bmu[1], distance)

    def get_node_data(
        self,
        node: Tuple[int, int],
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Get all samples assigned to a specific node.

        Args:
            node: Tuple of (x, y) coordinates.
            data: Data array (n_samples, n_features).

        Returns:
            Array of samples assigned to the node.

        Example:
            >>> node = (3, 7)
            >>> node_data = predictor.get_node_data(node, data)
            >>> print(f"Node has {len(node_data)} samples")
            Node has 5 samples
        """
        # Get assignments
        if isinstance(data, pd.DataFrame):
            data = data.values

        assignments = self._trainer.transform(data)
        mask = (assignments["neuron_x"] == node[0]) & (
            assignments["neuron_y"] == node[1]
        )
        return data[mask.values]

    def distance_map(self) -> np.ndarray:
        """
        Get the distance map (U-Matrix) as a 2D array.

        Returns:
            2D array where each cell contains the average distance
            to neighboring neurons.
        """
        return self._som.distance_map()

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using training parameters."""
        data_range = self._trainer._data_max - self._trainer._data_min
        data_range[data_range == 0] = 1
        return (x - self._trainer._data_min) / data_range
