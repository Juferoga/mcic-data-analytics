# Technical Design: SOM Integration

## Overview

This document describes the technical design for implementing Self-Organizing Maps (SOM) as the core unsupervised learning algorithm in the analitica library. SOM provides dimensionality reduction while preserving topological relationships, enabling clustering and visualization of high-dimensional data in ETL pipelines.

## Technical Approach

SOM integration follows the existing `BaseTransformer` pattern from the ETL module. The implementation wraps MiniSom 2.x and extends it with ETL-friendly interfaces for DataFrame processing, visualization, and quality analysis.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         analitica.som                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ SOMTrainer   │  │ SOMPredictor │  │ SOMVisualizer│          │
│  │              │  │              │  │              │          │
│  │ BaseTransform│  │ - winner()   │  │ - umatrix()  │          │
│  │ - init_*()  │  │ - quantize() │  │ - planes()   │          │
│  │ - train_*()  │  │ - error()     │  │ - bmu_plot() │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ SOMAnalyzer  │  │ SOMConfig    │  │ SOMException │          │
│  │              │  │              │  │              │          │
│  │ - QE, TE     │  │ Dataclass    │  │ - SOMError   │          │
│  │ - analyze()  │  │ Defaults     │  │ - NotTrained │          │
│  │ - report()   │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   MiniSom 2.x   │
                    │ (dependencies)   │
                    └─────────────────┘
```

---

## Architecture Decisions

### Decision: Why Normalize Data Before SOM

**Choice**: Require normalized input data (0-1 range recommended)
**Alternatives considered**: Allow raw data, auto-normalize internally
**Rationale**: SOM uses Euclidean distance for BMU finding. Raw data with different scales causes features with larger ranges to dominate the distance calculation. For example, if age (0-100) and income (0-100000) are both present, income would dominate the BMU selection. Pre-normalization ensures equal feature contribution and improves both convergence and interpretability.

### Decision: How Grid Size Affects Results

**Choice**: Default grid size = 5×√N (where N = number of samples), with bounds [3×3, 30×30]
**Alternatives considered**: Fixed 10×10, user-defined only
**Rationale**: Small maps (≤5×5) may merge distinct clusters; large maps (>20×20) have sparse neuron utilization. The 5√N rule provides a reasonable starting point. Bounds prevent memory issues (30×30 = 900 neurons × input_len weights) and ensure meaningful topology.

| Grid Size | Samples | Neurons | Use Case |
|-----------|---------|---------|----------|
| 5×5 | 100 | 25 | Quick prototyping |
| 10×10 | 500 | 100 | Standard clustering |
| 15×15 | 2000 | 225 | Detailed analysis |
| 20×20 | 10000 | 400 | Large datasets |

### Decision: Training Epochs vs Iterations

**Choice**: Use `iterations` (total weight updates) instead of `epochs` (passes over data)
**Alternatives considered**: Epochs-based training (MiniSom default)
**Rationale**: SOM training is stochastic; one "epoch" has different meaning depending on data order. Iterations provide explicit control over training effort. The proposal's 10-50×grid_size rule (e.g., 10×10 grid → 1000-5000 iterations) maps directly to effort, not data passes.

### Decision: Neighborhood Radius Decay Strategy

**Choice**: Exponential decay: σ(t) = σ₀ × exp(-t/τ)
**Alternatives considered**: Linear decay, step decay
**Rationale**: Exponential decay provides smooth, continuous transition from ordering phase (large σ) to tuning phase (small σ). This matches Kohonen's original algorithm and produces better topological preservation. The time constant τ ≈ max_iterations/4 ensures significant neighborhood size reduction by mid-training.

---

## Data Flow

### SOMTrainer Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Input DataFrame (pd.DataFrame)                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Columns: [feature_1, feature_2, ..., feature_n]            │  │
│  │ Shape: (n_samples, n_features)                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              ▼ fit()                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. Extract numeric columns to numpy array                │  │
│  │ 2. Initialize SOM weights (random/pca/samples)            │  │
│  │ 3. Train with configurable phases                         │  │
│  │ 4. Store _som, _weights, _normalizer                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              ▼ transform()                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Output: DataFrame with BMU coordinates                     │  │
│  │ Columns: [original_columns..., som_x, som_y]             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### SOMPredictor Data Flow

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  New Data        │───▶│  SOMPredictor    │───▶│  Results         │
│  (numpy array)   │    │  .winner()       │    │  (BMU coords)    │
└──────────────────┘    │  .quantize()     │    └──────────────────┘
                        │  .distance()     │
                        └──────────────────┘
```

---

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/analitica/som/trainer.py` | Create | SOMTrainer class (wraps MiniSom) |
| `src/analitica/som/predictor.py` | Create | SOMPredictor for post-training inference |
| `src/analitica/som/visualizer.py` | Create | Matplotlib-based U-matrix, component planes |
| `src/analitica/som/analyzer.py` | Create | Quality metrics (QE, TE) |
| `src/analitica/som/config.py` | Create | SOMConfig dataclass with defaults |
| `src/analitica/som/exceptions.py` | Create | SOMError, NotTrainedError |
| `src/analitica/som/__init__.py` | Modify | Export all new classes |
| `src/analitica/etl/transformer.py` | Modify | Optional: SOMTransformer for Pipeline |
| `tests/test_som.py` | Create | Unit and integration tests |
| `examples/demo_som.py` | Create | Usage examples |

---

## Implementation Details

### 1. SOMConfig (config.py)

```python
from dataclasses import dataclass, field
from typing import Tuple, Literal

@dataclass
class SOMConfig:
    """Configuration for SOM training and inference.
    
    Attributes:
        grid_size: Tuple of (width, height) neurons. Default (10, 10).
        input_len: Number of features in input vectors.
        sigma: Initial neighborhood radius. Default 1.0.
        learning_rate: Initial learning rate. Default 0.5.
        num_iterations: Total training iterations. Default 1000.
        random_seed: Reproducibility seed. Default None.
        init_method: Weight initialization. Options: 'random', 'pca', 'random_samples'.
        neighborhood_fn: Neighborhood function. Options: 'gaussian', 'mexican_hat', 'bubble', 'triangle'.
        decay_function: Learning rate decay. Options: 'asymptotic', 'inverse', 'linear'.
        verbose: Enable training progress output. Default False.
    """
    grid_size: Tuple[int, int] = (10, 10)
    input_len: int = 1
    sigma: float = 1.0
    learning_rate: float = 0.5
    num_iterations: int = 1000
    random_seed: int = None
    init_method: Literal['random', 'pca', 'random_samples'] = 'pca'
    neighborhood_fn: Literal['gaussian', 'mexican_hat', 'bubble', 'triangle'] = 'gaussian'
    decay_function: Literal['asymptotic', 'inverse', 'linear'] = 'asymptotic'
    verbose: bool = False
    
    def __post_init__(self):
        if self.grid_size[0] < 3 or self.grid_size[1] < 3:
            raise ValueError("Grid size must be at least 3×3")
        if self.grid_size[0] > 30 or self.grid_size[1] > 30:
            raise ValueError("Grid size cannot exceed 30×30")
```

### 2. SOMTrainer (trainer.py)

```python
import numpy as np
import pandas as pd
from minisom import MiniSom
from typing import Optional, List, Tuple

from analitica.etl.transformer import BaseTransformer
from analitica.som.config import SOMConfig
from analitica.som.exceptions import SOMError, NotTrainedError


class SOMTrainer(BaseTransformer):
    """Self-Organizing Map trainer with ETL interface.
    
    Wraps MiniSom with DataFrame input/output and stores
    normalization state for consistent transform() calls.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        num_iterations: int = 1000,
        random_seed: Optional[int] = None,
        init_method: str = 'pca',
        neighborhood_fn: str = 'gaussian',
        decay_function: str = 'asymptotic',
        columns: Optional[List[str]] = None,
    ):
        """Initialize SOM trainer.
        
        Args:
            grid_size: (width, height) of the SOM grid.
            sigma: Initial neighborhood radius.
            learning_rate: Initial learning rate.
            num_iterations: Number of training iterations.
            random_seed: Seed for reproducibility.
            init_method: Weight initialization method.
            neighborhood_fn: Neighborhood function type.
            decay_function: Learning rate decay type.
            columns: Columns to use. If None, all numeric columns.
        """
        self.config = SOMConfig(
            grid_size=grid_size,
            sigma=sigma,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            random_seed=random_seed,
            init_method=init_method,
            neighborhood_fn=neighborhood_fn,
            decay_function=decay_function,
        )
        self.columns = columns
        self._fitted_columns: List[str] = []
        self._som: Optional[MiniSom] = None
        self._weights: Optional[np.ndarray] = None
        self._normalizer_params: Optional[dict] = None
        self._is_fitted: bool = False
    
    def fit(self, data: pd.DataFrame) -> "SOMTrainer":
        """Train the SOM on input data.
        
        Args:
            data: Input DataFrame with numeric columns.
            
        Returns:
            self: The fitted trainer.
            
        Raises:
            SOMError: If data has insufficient samples or no numeric columns.
        """
        # Extract numeric columns
        columns = self.columns or self._get_numeric_columns(data)
        self._fitted_columns = [c for c in columns if c in data.columns]
        
        if not self._fitted_columns:
            raise SOMError("No numeric columns found for SOM training")
        
        # Convert to numpy and store normalization params
        X = data[self._fitted_columns].values.astype(np.float64)
        
        # Check for sufficient samples
        if len(X) < self.config.grid_size[0] * self.config.grid_size[1]:
            raise SOMError(
                f"Insufficient samples ({len(X)}) for grid size "
                f"{self.config.grid_size}"
            )
        
        # Store normalization params for transform
        self._normalizer_params = {
            'min': X.min(axis=0),
            'max': X.max(axis=0),
        }
        
        # Normalize to [0, 1] range (required for SOM)
        X_normalized = self._normalize(X)
        
        # Initialize SOM
        self.config.input_len = X.shape[1]
        self._som = MiniSom(
            x=self.config.grid_size[0],
            y=self.config.grid_size[1],
            input_len=self.config.input_len,
            sigma=self.config.sigma,
            learning_rate=self.config.learning_rate,
            random_seed=self.config.random_seed,
            neighborhood_function=self.config.neighborhood_fn,
            decay_function=self.config.decay_function,
        )
        
        # Initialize weights
        if self.config.init_method == 'pca':
            self._som.pca_weights_init(X_normalized)
        elif self.config.init_method == 'random_samples':
            self._som.random_weights_init(X_normalized)
        else:  # 'random'
            self._som.random_weights_init()
        
        # Train the SOM
        self._som.train_random(X_normalized, self.config.num_iterations)
        
        # Store weights and mark as fitted
        self._weights = self._som.get_weights()
        self._is_fitted = True
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by adding SOM coordinates.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            DataFrame with added som_x, som_y columns.
            
        Raises:
            NotTrainedError: If trainer has not been fitted.
        """
        if not self._is_fitted:
            raise NotTrainedError("SOMTrainer must be fitted before transform")
        
        result = data.copy()
        
        # Extract and normalize input columns
        X = data[self._fitted_columns].values.astype(np.float64)
        X_normalized = self._normalize(X)
        
        # Find BMU for each sample
        bmus = np.array([self._som.winner(x) for x in X_normalized])
        
        # Add coordinates
        result['som_x'] = bmus[:, 0]
        result['som_y'] = bmus[:, 1]
        
        return result
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range."""
        if self._normalizer_params is None:
            return X
        X_range = self._normalizer_params['max'] - self._normalizer_params['min']
        X_range[X_range == 0] = 1  # Avoid division by zero
        return (X - self._normalizer_params['min']) / X_range
    
    @staticmethod
    def _get_numeric_columns(data: pd.DataFrame) -> List[str]:
        """Get list of numeric columns."""
        return data.select_dtypes(include=[np.number]).columns.tolist()
    
    @property
    def som(self) -> MiniSom:
        """Get the underlying MiniSom instance."""
        if not self._is_fitted:
            raise NotTrainedError("SOMTrainer must be fitted first")
        return self._som
    
    @property
    def weights(self) -> np.ndarray:
        """Get the SOM weight matrix."""
        if not self._is_fitted:
            raise NotTrainedError("SOMTrainer must be fitted first")
        return self._weights
```

### 3. SOMPredictor (predictor.py)

```python
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

from analitica.som.exceptions import NotTrainedError


@dataclass
class PredictionResult:
    """Result of SOM prediction."""
    bmu_coords: Tuple[int, int]      # (x, y) coordinates of BMU
    quantization_error: float        # Distance to BMU
    activation_level: float          # Neuron activation (inverse distance)


class SOMPredictor:
    """Post-training predictions for SOM."""
    
    def __init__(self, som, weights: np.ndarray):
        """Initialize predictor with trained SOM.
        
        Args:
            som: Trained MiniSom instance.
            weights: SOM weight matrix from training.
        """
        self._som = som
        self._weights = weights
        self._grid_size = (som._weights.shape[0], som._weights.shape[1])
    
    def winner(self, x: np.ndarray) -> Tuple[int, int]:
        """Find Best Matching Unit for input vector.
        
        Args:
            x: Input vector (1D array).
            
        Returns:
            Tuple of (x_idx, y_idx) coordinates.
        """
        return self._som.winner(x)
    
    def winners(self, X: np.ndarray) -> List[Tuple[int, int]]:
        """Find BMUs for multiple input vectors.
        
        Args:
            X: Input matrix (2D array, samples × features).
            
        Returns:
            List of (x, y) coordinate tuples.
        """
        return [self._som.winner(x) for x in X]
    
    def quantization(self, X: np.ndarray) -> np.ndarray:
        """Get codebook vectors (BMU weights) for input data.
        
        Args:
            X: Input matrix.
            
        Returns:
            Matrix of codebook vectors.
        """
        bmus = self.winners(X)
        codebook = np.array([self._weights[bmu] for bmu in bmus])
        return codebook
    
    def quantization_error(self, X: np.ndarray) -> float:
        """Calculate average quantization error.
        
        Args:
            X: Input matrix.
            
        Returns:
            Average Euclidean distance to BMU.
        """
        errors = []
        for x in X:
            bmu = self._som.winner(x)
            bmu_weights = self._weights[bmu]
            error = np.linalg.norm(x - bmu_weights)
            errors.append(error)
        return np.mean(errors)
    
    def distance_to_bmu(self, x: np.ndarray) -> float:
        """Calculate Euclidean distance from input to its BMU.
        
        Args:
            x: Input vector.
            
        Returns:
            Distance to BMU.
        """
        bmu = self._som.winner(x)
        bmu_weights = self._weights[bmu]
        return np.linalg.norm(x - bmu_weights)
    
    def activation_response(self, X: np.ndarray) -> np.ndarray:
        """Count how often each neuron wins (hit histogram).
        
        Args:
            X: Input matrix.
            
        Returns:
            2D array of hit counts per neuron.
        """
        hits = np.zeros(self._grid_size)
        for x in X:
            bmu = self._som.winner(x)
            hits[bmu] += 1
        return hits
```

### 4. SOMVisualizer (visualizer.py)

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from analitica.som.exceptions import NotTrainedError


class SOMVisualizer:
    """Matplotlib-based visualizations for SOM."""
    
    def __init__(self, som, weights: np.ndarray):
        """Initialize visualizer with trained SOM.
        
        Args:
            som: Trained MiniSom instance.
            weights: SOM weight matrix.
        """
        self._som = som
        self._weights = weights
        self._grid_size = weights.shape[:2]
    
    def umatrix(
        self,
        figsize: Tuple[int, int] = (10, 10),
        cmap: str = 'jet',
        title: str = 'U-Matrix',
        show: bool = True,
    ) -> np.ndarray:
        """Generate U-Matrix (Unified Distance Matrix).
        
        The U-Matrix shows average distance between each neuron
        and its neighbors. High values indicate cluster boundaries.
        
        Args:
            figsize: Figure size in inches.
            cmap: Colormap name.
            title: Plot title.
            show: Whether to display the plot.
            
        Returns:
            2D array of U-Matrix values.
        """
        um = self._som.distance_matrix()
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(um, cmap=cmap, origin='lower')
        ax.set_title(title)
        ax.set_xlabel('SOM X')
        ax.set_ylabel('SOM Y')
        plt.colorbar(im, ax=ax, label='Distance')
        
        if show:
            plt.show()
        
        return um
    
    def component_planes(
        self,
        feature_names: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: str = 'jet',
        show: bool = True,
    ) -> np.ndarray:
        """Generate component planes for each feature.
        
        Component planes show how each input feature is distributed
        across the map. Correlated features have similar planes.
        
        Args:
            feature_names: Names for each feature dimension.
            figsize: Figure size (auto-calculated if None).
            cmap: Colormap name.
            show: Whether to display the plot.
            
        Returns:
            3D array of component plane values.
        """
        n_features = self._weights.shape[2]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (5 * n_cols, 5 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        planes = []
        for idx, (ax, name) in enumerate(zip(axes, feature_names)):
            # Extract plane for this feature
            plane = self._weights[:, :, idx]
            planes.append(plane)
            
            im = ax.imshow(plane, cmap=cmap, origin='lower')
            ax.set_title(name)
            ax.set_xlabel('SOM X')
            ax.set_ylabel('SOM Y')
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for ax in axes[n_features:]:
            ax.axis('off')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return np.array(planes)
    
    def bmu_highlight(
        self,
        data: np.ndarray,
        bmus: List[Tuple[int, int]],
        figsize: Tuple[int, int] = (10, 10),
        umatrix_cmap: str = 'gray',
        scatter_color: str = 'red',
        scatter_size: float = 50,
        title: str = 'BMU Locations',
        show: bool = True,
    ) -> None:
        """Overlay BMU locations on U-Matrix.
        
        Args:
            data: Original input data.
            bmu_coords: List of BMU coordinates.
            figsize: Figure size.
            umatrix_cmap: U-Matrix colormap.
            scatter_color: Color for BMU markers.
            scatter_size: Size of BMU markers.
            title: Plot title.
            show: Whether to display the plot.
        """
        um = self._som.distance_matrix()
        
        # Extract x, y coordinates
        x_coords = [bmu[0] for bmu in bmus]
        y_coords = [bmu[1] for bmu in bmus]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(um, cmap=umatrix_cmap, origin='lower')
        ax.scatter(x_coords, y_coords, c=scatter_color, s=scatter_size, 
                   alpha=0.7, edgecolors='black')
        ax.set_title(title)
        ax.set_xlabel('SOM X')
        ax.set_ylabel('SOM Y')
        
        if show:
            plt.show()
```

### 5. SOMAnalyzer (analyzer.py)

```python
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from analitica.som.exceptions import NotTrainedError


@dataclass
class QualityMetrics:
    """SOM quality metrics."""
    quantization_error: float
    topographic_error: float
    neuron_utilization: float
    hit_distribution: np.ndarray


class SOMAnalyzer:
    """Quality metrics calculation for SOM."""
    
    def __init__(self, som, weights: np.ndarray):
        """Initialize analyzer with trained SOM.
        
        Args:
            som: Trained MiniSom instance.
            weights: SOM weight matrix.
        """
        self._som = som
        self._weights = weights
        self._grid_size = (weights.shape[0], weights.shape[1])
    
    def quantization_error(self, X: np.ndarray) -> float:
        """Calculate quantization error (QE).
        
        QE measures how well data fits the map. Lower = better.
        
        Formula: QE = (1/N) × Σ ||x_i - w_BMU(x_i)||
        
        Args:
            X: Input data matrix.
            
        Returns:
            Average quantization error.
        """
        errors = []
        for x in X:
            bmu = self._som.winner(x)
            bmu_weights = self._weights[bmu]
            error = np.linalg.norm(x - bmu_weights)
            errors.append(error)
        return np.mean(errors)
    
    def topographic_error(self, X: np.ndarray, k: int = 2) -> float:
        """Calculate topographic error (TE).
        
        TE measures topology preservation. Lower = better.
        
        For each input, find top-k BMUs. TE = fraction where
        top-k neurons are NOT adjacent.
        
        Args:
            X: Input data matrix.
            k: Number of BMUs to consider (default 2).
            
        Returns:
            Topographic error (0 to 1).
        """
        errors = 0
        for x in X:
            # Get winners
            bmus = self._som.neuron_to_winner(x)
            if len(bmus) < k:
                continue
            
            # Check adjacency for top-k
            top_k = [b[1] for b in bmus[:k]]
            if not self._are_adjacent(top_k[0], top_k[1]):
                errors += 1
        
        return errors / len(X)
    
    def _are_adjacent(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Check if two neurons are adjacent (including diagonal)."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) <= 1
    
    def neuron_utilization(self, X: np.ndarray) -> float:
        """Calculate fraction of neurons that received hits.
        
        Args:
            X: Input data matrix.
            
        Returns:
            Utilization ratio (0 to 1).
        """
        hits = self.hit_histogram(X)
        utilized = np.sum(hits > 0)
        total = self._grid_size[0] * self._grid_size[1]
        return utilized / total
    
    def hit_histogram(self, X: np.ndarray) -> np.ndarray:
        """Calculate hit counts per neuron.
        
        Args:
            X: Input data matrix.
            
        Returns:
            2D array of hit counts.
        """
        hits = np.zeros(self._grid_size)
        for x in X:
            bmu = self._som.winner(x)
            hits[bmu] += 1
        return hits
    
    def analyze(self, X: np.ndarray) -> QualityMetrics:
        """Run full quality analysis.
        
        Args:
            X: Input data matrix.
            
        Returns:
            QualityMetrics dataclass.
        """
        return QualityMetrics(
            quantization_error=self.quantization_error(X),
            topographic_error=self.topographic_error(X),
            neuron_utilization=self.neuron_utilization(X),
            hit_distribution=self.hit_histogram(X),
        )
    
    def report(self, metrics: QualityMetrics) -> str:
        """Generate human-readable quality report.
        
        Args:
            metrics: QualityMetrics from analyze().
            
        Returns:
            Formatted report string.
        """
        lines = [
            "SOM Quality Report",
            "=" * 40,
            f"Quantization Error: {metrics.quantization_error:.4f}",
            f"Topographic Error:   {metrics.topographic_error:.4f}",
            f"Neuron Utilization: {metrics.neuron_utilization*100:.1f}%",
            "=" * 40,
        ]
        
        # Add interpretation
        if metrics.quantization_error < 0.1:
            lines.append("✓ Good data fit")
        elif metrics.quantization_error < 0.2:
            lines.append("~ Moderate data fit")
        else:
            lines.append("✗ Poor data fit - consider more iterations")
        
        if metrics.topographic_error < 0.05:
            lines.append("✓ Good topology preservation")
        elif metrics.topographic_error < 0.1:
            lines.append("~ Moderate topology")
        else:
            lines.append("✗ Poor topology - adjust sigma/neighborhood")
        
        return "\n".join(lines)
```

### 6. SOMExceptions (exceptions.py)

```python
"""Custom exceptions for SOM module."""


class SOMError(Exception):
    """Base exception for SOM-related errors."""
    pass


class NotTrainedError(SOMError):
    """Raised when operating on an untrained SOM.
    
    This exception indicates that SOMTrainer.fit() has not been
    called, or the SOM has not been properly initialized.
    """
    pass


class InvalidConfigurationError(SOMError):
    """Raised when SOM configuration is invalid."""
    pass


class InsufficientDataError(SOMError):
    """Raised when training data is insufficient for the grid size."""
    pass
```

---

## Interfaces / Contracts

### SOMTransformer (Optional ETL Integration)

For seamless Pipeline integration:

```python
class SOMTransformer(SOMTrainer):
    """SOM as a Pipeline transformer step."""
    
    def fit(self, data: pd.DataFrame, y=None) -> "SOMTransformer":
        return super().fit(data)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().transform(data)
```

---

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | SOMConfig validation | Test bounds, invalid configs |
| Unit | SOMTrainer.fit() | Mock MiniSom, verify initialization |
| Unit | SOMTrainer.transform() | Test BMU assignment |
| Unit | SOMPredictor methods | Test winner(), quantization() |
| Unit | SOMAnalyzer metrics | Compare to known values |
| Unit | SOMVisualizer plots | Test figure creation |
| Integration | Full pipeline | Train → Predict → Analyze → Visualize |
| E2E | Real data | Iris dataset or customer data |

### Test Data Fixtures

```python
@pytest.fixture
def som_trainer():
    """Provide a configured SOMTrainer."""
    return SOMTrainer(
        grid_size=(5, 5),
        num_iterations=100,
        random_seed=42,
    )

@pytest.fixture
def sample_data():
    """Provide synthetic 2D data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
    })
```

---

## Migration / Rollback

**No migration required.** This is a new module addition.

### Rollback Plan (if needed)
1. Delete `src/analitica/som/` files (trainer.py, predictor.py, visualizer.py, analyzer.py, config.py, exceptions.py)
2. Revert `src/analitica/som/__init__.py` to only export `SOMTrainer`
3. Remove `tests/test_som.py` and `examples/demo_som.py`
4. Uninstall `minisom` if not used elsewhere

---

## Open Questions

- [ ] Should SOMVisualizer support saving plots to files (PNG, SVG)?
- [ ] Should we add multiprocessing for large dataset training?
- [ ] Should we support lazy visualization (generate on demand)?
- [ ] Should we add cluster detection (e.g., K-means on SOM neurons)?

---

## Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|--------------|
| 1 | SOMTrainer extends BaseTransformer | Instance check |
| 2 | fit() trains SOM with all init methods | Unit tests |
| 3 | transform() adds correct BMU coords | Compare to manual calculation |
| 4 | SOMPredictor.winner() returns correct coords | Test known data |
| 5 | SOMVisualizer produces U-Matrix | Image comparison or dimensions |
| 6 | SOMAnalyzer computes QE correctly | Compare to formula |
| 7 | SOMConfig validates bounds | Test invalid configs raise errors |
| 8 | ETL Pipeline integration works | End-to-end test |

---

## Dependencies

### Required
- `minisom>=2.0` - Core SOM implementation (NEW)
- `numpy>=1.24.0` - Already in dependencies
- `pandas>=2.0.0` - Already in dependencies
- `matplotlib>=3.7.0` - Visualization (NEW)

### Update pyproject.toml
```toml
[project.optional-dependencies]
som = ["minisom>=2.0", "matplotlib>=3.7.0"]
```
