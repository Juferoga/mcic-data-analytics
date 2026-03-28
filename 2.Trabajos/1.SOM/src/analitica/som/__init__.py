"""
SOM Module - Self-Organizing Maps integration.

This module provides tools for training, predicting, visualizing,
and analyzing Self-Organizing Maps (SOMs).

Self-Organizing Maps (SOM) or Kohonen Maps
==========================================

A SOM is an unsupervised learning algorithm that produces a low-dimensional
(typically 2D) representation of higher-dimensional data while preserving
topological relationships.

Key Concepts:
-------------
1. Grid of Neurons: The SOM consists of a grid of neurons, each with
   a weight vector of the same dimension as the input data.

2. Best Matching Unit (BMU): For each input vector, the neuron with
   the most similar weight vector is the BMU.

3. Neighborhood Function: The BMU and its neighbors are updated to be
   more similar to the input. The influence decreases with distance.

4. Training Process:
   - Initialization: Weight vectors are initialized (randomly or via PCA)
   - Ordering Phase: Large neighborhood, high learning rate (rough tuning)
   - Tuning Phase: Small neighborhood, low learning rate (fine-tuning)

Mathematical Formulation:
-------------------------
For an input vector x, the BMU b is found as:
    b = argmin_i ||x - w_i||

Weight updates:
    w_i(t+1) = w_i(t) + h_ci(t) * alpha(t) * (x - w_i(t))

Where:
- h_ci(t): Neighborhood function (usually Gaussian)
- alpha(t): Learning rate at time t
- t: Training iteration

Usage Example:
--------------
    from analitica.som import SOMTrainer

    # Prepare normalized data
    data = df[['feature1', 'feature2', 'feature3']].values

    # Create and train SOM
    trainer = SOMTrainer(x=10, y=10, input_len=3)
    trainer.fit(data, epochs=100)

    # Get neuron assignments
    assignments = trainer.transform(data)
"""

from .trainer import SOMTrainer
from .predictor import SOMPredictor
from .visualizer import SOMVisualizer
from .analyzer import SOMAnalyzer
from .config import SOMConfig
from .exceptions import (
    SOMError,
    NotTrainedError,
    InvalidConfigurationError,
    InsufficientDataError,
)

__all__ = [
    "SOMTrainer",
    "SOMPredictor",
    "SOMVisualizer",
    "SOMAnalyzer",
    "SOMConfig",
    "SOMError",
    "NotTrainedError",
    "InvalidConfigurationError",
    "InsufficientDataError",
]
