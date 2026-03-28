"""
SOM Configuration dataclass.

This module defines the SOMConfig class that holds all parameters
needed to create and train a Self-Organizing Map.

Understanding these parameters is crucial for effective SOM usage.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SOMConfig:
    """
    Configuration for Self-Organizing Map training.

    This dataclass encapsulates all parameters needed to initialize
    and train a SOM. Default values are chosen for general-purpose use,
    but should be tuned for specific applications.

    Attributes:
        x: Width of the grid in neurons (columns).
            Typical range: 3-30. Rule of thumb: 5*sqrt(N) where N is sample count.
            Larger grids can capture more detail but require more training.

        y: Height of the grid in neurons (rows).
            Typical range: 3-30. Same considerations as x.

        input_len: Dimensionality of input vectors (number of features).
            Must match the number of features in your data.
            Example: If data has 10 features, input_len=10.

        sigma: Initial radius of the neighborhood function.
            Typical range: 1.0 to max(x, y)/2.
            Controls how many neurons are affected during weight updates.
            Larger sigma = more global structure learned first.
            Default: 1.0

        learning_rate: Initial learning rate for weight updates.
            Typical range: 0.01 to 0.5.
            Controls how much weights change during updates.
            Larger values = faster initial learning.
            Default: 0.5

        random_seed: Random seed for reproducibility.
            Set to an integer for reproducible results.
            Set to None for random initialization.
            Default: 42

        neighborhood_function: Function that defines neighborhood influence.
            Options:
            - 'gaussian': Smooth Gaussian decay (most common, smooth transitions)
            - 'mexican_hat': Mexican hat wavelet (sharper boundaries)
            - 'bubble': Flat disc with sharp edge
            - 'triangle': Linear decay to zero
            Default: 'gaussian'

        initialization: How initial weights are set.
            Options:
            - 'random': Random initialization (default)
            - 'pca': Initialize along PCA axes (faster convergence)
            - 'random_samples': Initialize from random data samples
            Default: 'random'

    Example:
        # Small grid for quick prototyping
        config = SOMConfig(x=5, y=5, input_len=4)

        # Larger grid for detailed analysis
        config = SOMConfig(x=15, y=15, input_len=10, sigma=2.0)

        # With PCA initialization for faster convergence
        config = SOMConfig(x=10, y=10, initialization='pca')

    Mathematical Notes:
        - The neighborhood function determines how influence spreads from the BMU
        - sigma controls the initial neighborhood radius
        - learning_rate controls the step size in weight space
        - Both sigma and learning_rate decay during training
    """

    x: int = 10
    y: int = 10
    input_len: int = 1
    sigma: float = 1.0
    learning_rate: float = 0.5
    random_seed: Optional[int] = 42
    neighborhood_function: str = "gaussian"
    initialization: str = "random"

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        from .exceptions import InvalidConfigurationError

        if self.x <= 0:
            raise InvalidConfigurationError(
                f"Grid width (x) must be positive, got {self.x}"
            )
        if self.y <= 0:
            raise InvalidConfigurationError(
                f"Grid height (y) must be positive, got {self.y}"
            )
        if self.input_len <= 0:
            raise InvalidConfigurationError(
                f"Input length must be positive, got {self.input_len}"
            )
        if self.sigma <= 0:
            raise InvalidConfigurationError(f"Sigma must be positive, got {self.sigma}")
        if not 0 < self.learning_rate <= 1:
            raise InvalidConfigurationError(
                f"Learning rate must be in (0, 1], got {self.learning_rate}"
            )
        if self.neighborhood_function not in [
            "gaussian",
            "mexican_hat",
            "bubble",
            "triangle",
        ]:
            raise InvalidConfigurationError(
                f"Unknown neighborhood function: {self.neighborhood_function}. "
                f"Options: 'gaussian', 'mexican_hat', 'bubble', 'triangle'"
            )
        if self.initialization not in ["random", "pca", "random_samples"]:
            raise InvalidConfigurationError(
                f"Unknown initialization: {self.initialization}. "
                f"Options: 'random', 'pca', 'random_samples'"
            )
