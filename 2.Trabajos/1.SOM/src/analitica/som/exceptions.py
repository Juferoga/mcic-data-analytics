"""
SOM-specific exceptions.

This module defines custom exceptions used throughout the SOM module.
Proper exception handling helps users understand when SOM operations fail
and why.
"""


class SOMError(Exception):
    """Base exception for SOM-related errors.

    All SOM-specific exceptions should inherit from this class.
    This allows users to catch all SOM errors with a single except clause.

    Example:
        try:
            trainer.fit(data)
        except SOMError as e:
            print(f"SOM error: {e}")
    """

    pass


class NotTrainedError(SOMError):
    """Raised when trying to use an untrained SOM.

    The SOM must be trained (via fit() or fit_transform()) before
    making predictions or visualizations.

    Example:
        predictor = SOMPredictor()  # Not trained yet
        predictor.bmu(sample)  # Raises NotTrainedError
    """

    pass


class InvalidConfigurationError(SOMError):
    """Raised when SOM configuration parameters are invalid.

    Common causes:
    - Negative grid dimensions
    - Learning rate out of valid range
    - Invalid neighborhood function name

    Example:
        config = SOMConfig(x=0, y=10)  # x=0 is invalid
        # Raises InvalidConfigurationError
    """

    pass


class InsufficientDataError(SOMError):
    """Raised when there's not enough data for SOM training.

    A SOM needs sufficient samples to learn meaningful patterns.
    As a rule of thumb, you need at least 3 * x * y samples.

    Example:
        data = np.random.rand(5, 10)  # Only 5 samples
        trainer = SOMTrainer(x=5, y=5)  # Needs 75+ samples
        trainer.fit(data)  # Raises InsufficientDataError
    """

    pass
