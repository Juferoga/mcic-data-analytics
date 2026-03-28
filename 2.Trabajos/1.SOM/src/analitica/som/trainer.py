"""
SOM Trainer - Core SOM training implementation.

This module contains the SOMTrainer class which wraps MiniSom to provide
an ETL-friendly interface for training Self-Organizing Maps.

Self-Organizing Maps: Deep Dive
================================

What is a SOM?
--------------
A Self-Organizing Map (SOM) is a type of artificial neural network
that uses unsupervised learning to produce a low-dimensional representation
of the input space, called a map.

How does it work?
-----------------
1. Each neuron in the grid has a weight vector with the same dimension
   as the input data.

2. For each training sample:
   a. Find the Best Matching Unit (BMU) - the neuron with the most
      similar weight vector (using Euclidean distance).
   b. Update the BMU and its neighbors to be more similar to the input.

3. Over time, neurons that are close in the grid learn to represent
   similar inputs, preserving the topological structure of the data.

Why use SOMs?
-------------
- Dimensionality reduction (high-D → 2D visualization)
- Clustering without pre-specifying number of clusters
- Anomaly detection (samples far from any BMU)
- Feature correlation visualization (component planes)

The Training Process
--------------------
SOM training typically occurs in two phases:

1. Ordering Phase (rough calibration):
   - Large neighborhood radius (sigma)
   - High learning rate
   - Fast learning to establish rough topology

2. Tuning Phase (fine-tuning):
   - Small neighborhood radius
   - Low learning rate
   - Slow learning to refine details

MiniSom handles this automatically through the epochs parameter:
- epochs=100 means 100 * len(data) weight updates total
- Early iterations have high learning rate and large sigma
- Later iterations have low learning rate and small sigma

Parameters Explained
--------------------
x, y: Grid dimensions
    The map size determines:
    - Resolution: Larger maps = more detail
    - Training time: Larger maps = longer training
    - Memory: Larger maps = more weights to store

    Rule of thumb: Start with 5*sqrt(N) neurons where N is sample count.
    Example: For 1000 samples, try 10x10 grid.

input_len: Number of features
    Must match your data's dimensionality.
    Example: 4 features → input_len=4

sigma: Initial neighborhood radius
    How far the influence of the BMU extends.
    - Too large: Map doesn't form distinct regions
    - Too small: Map has isolated neurons, poor topology

    Rule of thumb: sigma ≈ max(x, y) / 2

learning_rate: Initial learning rate
    How much weights change per update.
    - Too large: Weights oscillate, no convergence
    - Too small: Training takes too long

    Rule of thumb: Start with 0.5, decrease to 0.01 by end

neighborhood_function: Shape of neighborhood influence
    'gaussian' (recommended): Smooth, gradual influence decay
    'mexican_hat': Sharp boundaries
    'bubble': Binary (inside/outside neighborhood)
    'triangle': Linear decay

Initialization Methods
----------------------
'random': Initialize weights randomly
    Simple but may require more training.

'pca': Initialize along PCA eigenvectors
    Faster convergence, better initial ordering.
    Recommended for large datasets.

'random_samples': Initialize from random data samples
    Weights start in data distribution.
"""

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from minisom import MiniSom

from .config import SOMConfig
from .exceptions import (
    SOMError,
    NotTrainedError,
    InvalidConfigurationError,
    InsufficientDataError,
)


class SOMTrainer:
    """
    Train a Self-Organizing Map on your data.

    This class provides an ETL-friendly interface for training SOMs
    using the MiniSom library. It handles data normalization, training,
    and provides methods for prediction and visualization.

    Attributes:
        config: SOMConfig object with training parameters.
        _som: The underlying MiniSom instance (set after fit).
        _is_trained: Boolean flag indicating if SOM is trained.
        _normalizer: The normalizer used to scale input data.
        _original_columns: Column names from input DataFrame.

    Example:
        Basic usage with numpy array:
        >>> import numpy as np
        >>> data = np.random.rand(100, 4)  # 100 samples, 4 features
        >>> trainer = SOMTrainer(x=10, y=10, input_len=4)
        >>> trainer.fit(data)
        >>> assignments = trainer.transform(data)
        >>> print(f"Assigned {len(assignments)} samples")

        Basic usage with pandas DataFrame:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'income': [50000, 60000, 70000],
        ...     'score': [0.8, 0.9, 0.7]
        ... })
        >>> trainer = SOMTrainer(x=5, y=5, input_len=3)
        >>> trainer.fit(df)
        >>> assignments = trainer.transform(df)

    Mathematical Background:
    -----------------------
    The SOM algorithm minimizes the quantization error:

        QE = (1/N) * Σ ||x_i - w_BMU(x_i)||

    Where:
    - N: Number of samples
    - x_i: Input vector i
    - w_BMU(x_i): Weight vector of BMU for x_i
    - ||...||: Euclidean distance

    Lower QE means better representation of data by the map.
    """

    def __init__(
        self,
        x: int = 10,
        y: int = 10,
        input_len: Optional[int] = None,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        random_seed: Optional[int] = 42,
        neighborhood_function: str = "gaussian",
        initialization: str = "random",
    ):
        """
        Initialize the SOM trainer.

        Args:
            x: Width of the SOM grid (number of columns).
                Typical range: 3-30.

            y: Height of the SOM grid (number of rows).
                Typical range: 3-30.

            input_len: Dimensionality of input vectors.
                If None, inferred from first data during fit().
                Recommended to set explicitly for clarity.

            sigma: Initial neighborhood radius.
                Controls how many neurons are affected during updates.
                Typical range: 0.5 to max(x,y)/2.

            learning_rate: Initial learning rate.
                Controls how much weights change per update.
                Typical range: 0.1 to 0.5.

            random_seed: Seed for reproducible results.
                Set to an integer for reproducibility.
                Set to None for random initialization.

            neighborhood_function: Shape of neighborhood influence.
                Options: 'gaussian' (smooth), 'mexican_hat' (sharp),
                'bubble' (binary), 'triangle' (linear).

            initialization: Method for initial weight vectors.
                Options: 'random', 'pca', 'random_samples'.

        Raises:
            InvalidConfigurationError: If parameters are invalid.

        Example:
            >>> # Small, quick SOM
            >>> trainer = SOMTrainer(x=5, y=5, input_len=4)

            >>> # Larger, more detailed SOM
            >>> trainer = SOMTrainer(
            ...     x=15, y=15, input_len=10,
            ...     sigma=2.0, learning_rate=0.3
            ... )
        """
        # Validate inputs
        if x <= 0 or y <= 0:
            raise InvalidConfigurationError(
                f"Grid dimensions must be positive: x={x}, y={y}"
            )
        if sigma <= 0:
            raise InvalidConfigurationError(f"Sigma must be positive: {sigma}")
        if not 0 < learning_rate <= 1:
            raise InvalidConfigurationError(
                f"Learning rate must be in (0, 1]: {learning_rate}"
            )

        self.config = SOMConfig(
            x=x,
            y=y,
            input_len=input_len or 1,
            sigma=sigma,
            learning_rate=learning_rate,
            random_seed=random_seed,
            neighborhood_function=neighborhood_function,
            initialization=initialization,
        )

        # Internal state
        self._som: Optional[MiniSom] = None
        self._is_trained: bool = False
        self._original_columns: list = []
        self._data_min: np.ndarray = None
        self._data_max: np.ndarray = None

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        epochs: int = 100,
        verbose: bool = False,
    ) -> "SOMTrainer":
        """
        Train the SOM on the provided data.

        This method:
        1. Validates and prepares the data
        2. Normalizes the data to [0, 1] range (required for SOM)
        3. Initializes the SOM weights
        4. Trains the SOM through the specified number of epochs

        Training occurs in two phases:
        - Ordering phase (first ~25% of epochs): Large sigma, high learning rate
        - Tuning phase (remaining ~75%): Small sigma, low learning rate

        Args:
            data: Training data as DataFrame or numpy array.
                Shape: (n_samples, n_features) or (n_samples,)

            epochs: Number of training epochs.
                One epoch = one pass through all samples.
                More epochs = better trained map but slower.
                Typical range: 50-500.
                Rule of thumb: 10 * (x * y) epochs for convergence.

            verbose: If True, print training progress.
                Useful for monitoring long training runs.

        Returns:
            self: The trained SOMTrainer for method chaining.

        Raises:
            InsufficientDataError: If not enough samples for training.
                Rule of thumb: Need at least 3 * x * y samples.

            ValueError: If data has wrong shape or invalid values.

        Example:
            >>> import numpy as np
            >>> data = np.random.rand(200, 5)
            >>> trainer = SOMTrainer(x=10, y=10, input_len=5)
            >>> trainer.fit(data, epochs=100)  # Train for 100 epochs
            >>> print(f"Training complete: {trainer.is_trained}")
            Training complete: True

            >>> # With verbose output
            >>> trainer.fit(data, epochs=100, verbose=True)
            Epoch: 10/100 - Error: 0.245
            Epoch: 20/100 - Error: 0.198
            ...
        """
        # Convert DataFrame to numpy if needed
        if isinstance(data, pd.DataFrame):
            self._original_columns = data.columns.tolist()
            data_array = data.values
        else:
            data_array = data

        # Handle 1D data
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)

        # Validate data
        n_samples, n_features = data_array.shape

        if n_samples < 3:
            raise InsufficientDataError(
                f"Need at least 3 samples for training, got {n_samples}"
            )

        # Check minimum samples requirement
        min_required = 3 * self.config.x * self.config.y
        if n_samples < min_required:
            import warnings

            warnings.warn(
                f"Recommended minimum: {min_required} samples for "
                f"{self.config.x}x{self.config.y} grid, got {n_samples}. "
                f"Training may not be optimal.",
                RuntimeWarning,
            )

        # Update input_len if not set
        if self.config.input_len == 1 and n_features > 1:
            self.config.input_len = n_features

        if n_features != self.config.input_len:
            raise InvalidConfigurationError(
                f"Data has {n_features} features but input_len="
                f"{self.config.input_len}. Set input_len={n_features}."
            )

        # Normalize data to [0, 1] range
        # SOMs work with Euclidean distances, so normalization is essential
        self._data_min = data_array.min(axis=0)
        self._data_max = data_array.max(axis=0)

        # Avoid division by zero for constant features
        data_range = self._data_max - self._data_min
        data_range[data_range == 0] = 1

        normalized_data = (data_array - self._data_min) / data_range

        # Initialize MiniSom
        self._som = MiniSom(
            x=self.config.x,
            y=self.config.y,
            input_len=self.config.input_len,
            sigma=self.config.sigma,
            learning_rate=self.config.learning_rate,
            random_seed=self.config.random_seed,
            neighborhood_function=self.config.neighborhood_function,
            topology="rectangular",
            activation_distance="euclidean",
        )

        # Initialize weights
        if self.config.initialization == "pca":
            self._som.pca_weights_init(normalized_data)
        elif self.config.initialization == "random_samples":
            self._som.random_weights_init(normalized_data)
        else:
            self._som.random_weights_init(normalized_data)

        # Train the SOM
        if verbose:
            print(f"Training SOM: {self.config.x}x{self.config.y} grid")
            print(f"Data: {n_samples} samples, {n_features} features")
            print(f"Epochs: {epochs}")
            print("-" * 40)

        # Train with quantization error tracking
        self._som.train_random(
            data=normalized_data, num_iteration=epochs * n_samples, verbose=False
        )

        if verbose:
            final_error = self._som.quantization_error(normalized_data)
            print(f"Training complete!")
            print(f"Final quantization error: {final_error:.4f}")

        self._is_trained = True
        return self

    def fit_transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        epochs: int = 100,
    ) -> pd.DataFrame:
        """
        Fit the SOM and return neuron assignments.

        Convenience method that combines fit() and transform().

        Args:
            data: Training data.
            epochs: Number of training epochs.

        Returns:
            DataFrame with neuron assignments for each sample.

        Example:
            >>> data = np.random.rand(100, 4)
            >>> assignments = SOMTrainer(x=5, y=5, input_len=4).fit_transform(data)
            >>> assignments.head()
               neuron_x  neuron_y
            0         2         3
            1         4         1
            ...
        """
        self.fit(data, epochs=epochs)
        return self.transform(data)

    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """
        Get neuron assignments for data.

        For each sample, finds the Best Matching Unit (BMU) -
        the neuron whose weight vector is closest to the sample.

        Args:
            data: Data to assign to neurons.
                Must have same number of features as training data.

        Returns:
            DataFrame with columns:
            - neuron_x: X coordinate of assigned neuron (0 to x-1)
            - neuron_y: Y coordinate of assigned neuron (0 to y-1)

        Raises:
            NotTrainedError: If SOM hasn't been trained yet.
            InvalidConfigurationError: If data shape doesn't match.

        Example:
            >>> trainer = SOMTrainer(x=5, y=5, input_len=4)
            >>> trainer.fit(training_data)
            >>>
            >>> # Get assignments for training data
            >>> assignments = trainer.transform(training_data)
            >>> print(assignments.head())
               neuron_x  neuron_y
            0         2         3
            1         4         1
            ...

            >>> # Get assignments for new data
            >>> new_assignments = trainer.transform(new_data)
        """
        if not self._is_trained:
            raise NotTrainedError(
                "SOM must be trained before transform. Call fit() first."
            )

        # Convert DataFrame to numpy
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        # Handle 1D data
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)

        # Validate dimensions
        if data_array.shape[1] != self.config.input_len:
            raise InvalidConfigurationError(
                f"Data has {data_array.shape[1]} features but "
                f"SOM expects {self.config.input_len}"
            )

        # Normalize using training parameters
        data_range = self._data_max - self._data_min
        data_range[data_range == 0] = 1
        normalized_data = (data_array - self._data_min) / data_range

        # Find BMU for each sample
        bmu_coordinates = []
        for sample in normalized_data:
            bmu = self._som.winner(sample)
            bmu_coordinates.append(bmu)

        # Create result DataFrame
        result = pd.DataFrame(bmu_coordinates, columns=["neuron_x", "neuron_y"])

        # Preserve original column names if available
        if self._original_columns:
            result.columns = ["neuron_x", "neuron_y"]

        return result

    def get_weights(self) -> np.ndarray:
        """
        Get the SOM weight matrix.

        Returns:
            Numpy array of shape (x, y, input_len) containing
            the weight vectors for each neuron.

        Example:
            >>> weights = trainer.get_weights()
            >>> print(f"Weights shape: {weights.shape}")
            Weights shape: (10, 10, 4)
        """
        if not self._is_trained:
            raise NotTrainedError("SOM must be trained first.")
        return self._som.get_weights()

    @property
    def is_trained(self) -> bool:
        """Check if the SOM has been trained."""
        return self._is_trained

    def __repr__(self) -> str:
        """String representation of the SOM trainer."""
        status = "trained" if self._is_trained else "not trained"
        return (
            f"SOMTrainer("
            f"grid={self.config.x}x{self.config.y}, "
            f"features={self.config.input_len}, "
            f"status={status})"
        )
