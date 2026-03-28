# SOM (Self-Organizing Maps) Integration Specification

## Purpose

This specification defines the Self-Organizing Maps (SOM) module for the analitica project. SOM provides unsupervised dimensionality reduction while preserving topological relationships, enabling clustering and visualization of high-dimensional data in ETL pipelines.

## Scope

This is a NEW module specification. No existing specs are modified.

---

# SOMTrainer Specification

## Class: SOMTrainer

**File**: `src/analitica/som/trainer.py`

**Purpose**: Initialize, train, and store a Self-Organizing Map with flexible configuration options.

### Mathematical Foundation

A SOM is a grid of neurons where each neuron `w_ij` has a weight vector `w_ij ∈ ℝ^d` (same dimension as input data). Training uses competitive learning:

```
BMU(x) = argmin_{i,j} ||x - w_ij||  // Best Matching Unit
w_ij(t+1) = w_ij(t) + α(t) · h_{ij,BMU}(t) · (x - w_ij(t))
```

Where:
- `α(t)` = learning rate at iteration t (decays over time)
- `h_{ij,BMU}(t)` = neighborhood function (Gaussian: exp(-d²/(2σ²)))
- `d` = Manhattan or Euclidean distance from BMU to neuron (i,j)

### Method: __init__

```python
def __init__(
    self,
    x: int = 10,
    y: int = 10,
    input_len: int,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    random_seed: int = 42,
    neighborhood_function: str = "gaussian",
    initialization: str = "random"
) -> None:
```

**Parameters**:

| Parameter | Type | Default | Description | Typical Values |
|-----------|------|---------|-------------|----------------|
| `x` | int | 10 | Width of the SOM grid (number of columns) | 5-30, depends on data complexity |
| `y` | int | 10 | Height of the SOM grid (number of rows) | 5-30, depends on data complexity |
| `input_len` | int | Required | Dimension of input vectors (d) | Match your feature count |
| `sigma` | float | 1.0 | Initial neighborhood radius (σ) | 0.5-2.0, decrease for fine-tuning |
| `learning_rate` | float | 0.5 | Initial learning rate (α₀) | 0.3-0.7 for ordering, 0.05 for tuning |
| `random_seed` | int | 42 | Random seed for reproducibility | Any integer |
| `neighborhood_function` | str | "gaussian" | Neighborhood function type | "gaussian", "mexican_hat", "bubble", "triangle" |
| `initialization` | str | "random" | Weight initialization method | "random", "pca", "random_samples" |

**Grid Size Rule of Thumb**: `grid_size = 5 × √N` where N = number of samples. For 1000 samples: 5 × √1000 ≈ 158 neurons, so roughly 13×12 grid.

**Docstring Example**:

```python
"""
Self-Organizing Map Trainer.

A SOM is an unsupervised neural network that maps high-dimensional input
vectors onto a lower-dimensional (typically 2D) grid while preserving
topological relationships. Similar inputs activate neurons that are
spatially close on the map.

Mathematical Foundation:
    Given input vector x ∈ ℝ^d and neuron weight vectors w_ij ∈ ℝ^d:
    
    1. Find Best Matching Unit (BMU):
       BMU(x) = argmin_{i,j} ||x - w_ij||₂
       
    2. Update weights (Competitive Learning):
       w_ij(t+1) = w_ij(t) + α(t) · h_ij(t) · (x - w_ij(t))
       
    Where α(t) is the learning rate and h_ij(t) is the neighborhood function.

Example:
    >>> import numpy as np
    >>> from analitica.som import SOMTrainer
    >>> 
    >>> # Generate sample data (100 samples, 4 features)
    >>> data = np.random.rand(100, 4)
    >>> 
    >>> # Initialize and train SOM
    >>> som = SOMTrainer(x=10, y=10, input_len=4, random_seed=42)
    >>> som.fit(data, epochs=5000)
    >>> 
    >>> # Transform new data to node coordinates
    >>> coords = som.transform(data[:5])
    >>> print(coords)  # array of (x, y) tuples
"""
```

### Method: fit

```python
def fit(self, data: np.ndarray, epochs: int = 10000, verbose: bool = True) -> "SOMTrainer":
```

**Purpose**: Initialize the SOM grid with data and train it.

**Mathematical Description**: The fit method performs two training phases:

1. **Ordering Phase (first ~20% of epochs)**:
   - Large learning rate (typically α₀)
   - Large neighborhood radius (σ = max(x, y) / 2)
   - Establishes rough topology

2. **Tuning Phase (remaining ~80% of epochs)**:
   - Decreased learning rate (α → α × 0.01)
   - Decreased neighborhood radius (σ → 0.01)
   - Fine-tunes precise mappings

**Parameters**:

| Parameter | Type | Description | Typical Values |
|-----------|------|-------------|----------------|
| `data` | np.ndarray | Training data of shape (n_samples, input_len) | Normalized data in range [0, 1] |
| `epochs` | int | Number of training iterations | 1000-50000, depends on data size |
| `verbose` | bool | Print progress every 10% | True for debugging |

**Returns**: Self (for method chaining).

**Data Normalization Requirement**: The method MUST normalize input data to [0, 1] range using min-max scaling before training. Store normalization parameters for inverse transform.

**Docstring Example**:

```python
"""
Train the Self-Organizing Map on input data.

This method performs both the initialization of the SOM grid (if not already
initialized) and the competitive learning process that adjusts neuron weights
to match the distribution of the input data.

The training process:
    1. Initializes weight vectors using specified method (random/pca/samples)
    2. Ordering phase: ~20% of epochs with high learning rate
       - Establishes global topology
       - Learning rate: α₀ → α₀ × 0.01
       - Neighborhood: σ₀ → σ₀ × 0.01
    3. Tuning phase: ~80% of epochs with low learning rate
       - Refines local relationships
       - Learning rate: α₀ × 0.01 → 0.001
       - Neighborhood: σ₀ × 0.01 → 0.01

Parameters:
    data: np.ndarray of shape (n_samples, input_len)
        Training samples. MUST be normalized to [0, 1] range.
        Example: [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], ...]
    epochs: int, default 10000
        Total training iterations. Rule: 10-50 × grid_size²
        For 10×10 grid: 1000-5000 epochs minimum
    verbose: bool, default True
        If True, prints progress every 10% of epochs

Returns:
    self: SOMTrainer instance for method chaining

Raises:
    ValueError: If data.shape[1] != input_len
    ValueError: If data contains NaN or inf values

Example:
    >>> import numpy as np
    >>> from analitica.som import SOMTrainer
    >>> 
    >>> # Create normalized data (min-max scaled to [0,1])
    >>> data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    >>> 
    >>> # Train with 5000 epochs
    >>> som = SOMTrainer(x=5, y=5, input_len=3)
    >>> som.fit(data, epochs=5000, verbose=False)
    >>> 
    >>> # Check training history
    >>> print(f"Final QE: {som.quantization_error[-1]:.4f}")
"""
```

### Method: transform

```python
def transform(self, data: np.ndarray) -> np.ndarray:
```

**Purpose**: Map input vectors to their Best Matching Unit coordinates on the trained SOM grid.

**Mathematical Description**: For each input vector x ∈ ℝ^d, find the neuron (i, j) whose weight vector w_ij is closest:

```
BMU(x) = argmin_{i,j} ||x - w_ij||₂
```

Returns the (x_coord, y_coord) of the winning neuron for each input sample.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | np.ndarray | Input data of shape (n_samples, input_len) |

**Returns**: np.ndarray of shape (n_samples, 2) containing (x, y) coordinates for each sample.

**Docstring Example**:

```python
"""
Transform data to SOM node assignments.

Maps each input vector to its Best Matching Unit (BMU) on the trained SOM grid.
The BMU is the neuron whose weight vector is most similar to the input vector.

Mathematical Definition:
    For input vector x ∈ ℝ^d:
        BMU(x) = argmin_{i,j} ||x - w_ij||₂
    
    where w_ij is the weight vector of neuron at position (i, j).

Parameters:
    data: np.ndarray of shape (n_samples, input_len)
        Input vectors to map. Should be normalized to [0, 1].
        If not normalized, uses stored normalization parameters.

Returns:
    np.ndarray of shape (n_samples, 2)
        Each row contains [x_coord, y_coord] of the BMU.
        Coordinates are integers in range [0, x-1] × [0, y-1].

Example:
    >>> som = SOMTrainer(x=10, y=10, input_len=4)
    >>> som.fit(training_data, epochs=5000)
    >>> 
    >>> # Transform new samples
    >>> new_samples = np.random.rand(3, 4)
    >>> coords = som.transform(new_samples)
    >>> print(coords)
    [[3 7]
     [1 2]
     [9 4]]
    
    # Each row shows where each sample maps on the 10×10 grid
"""
```

### Method: fit_transform

```python
def fit_transform(self, data: np.ndarray, epochs: int = 10000) -> np.ndarray:
```

**Purpose**: Convenience method that combines fit() and transform() in a single call.

**Description**: Equivalent to calling `fit(data, epochs)` followed by `transform(data)`.

**Returns**: np.ndarray of shape (n_samples, 2) with BMU coordinates.

**Docstring Example**:

```python
"""
Train SOM and transform data in one step.

This is a convenience method equivalent to:
    som.fit(data, epochs)
    return som.transform(data)

Useful for quick experimentation and pipelines that need
to fit and transform in a single call.

Parameters:
    data: np.ndarray of shape (n_samples, input_len)
        Training data (will be normalized if not already)
    epochs: int, default 10000
        Number of training epochs

Returns:
    np.ndarray of shape (n_samples, 2)
        BMU coordinates for each input sample

Example:
    >>> data = np.random.rand(200, 5)
    >>> som = SOMTrainer(x=10, y=10, input_len=5)
    >>> coords = som.fit_transform(data, epochs=5000)
    >>> coords.shape
    (200, 2)
"""
```

### Method: get_weights

```python
def get_weights(self) -> np.ndarray:
```

**Purpose**: Return the current weight matrix of the SOM.

**Returns**: np.ndarray of shape (x, y, input_len) - the codebook vectors.

### Method: save

```python
def save(self, path: str) -> None:
```

**Purpose**: Save trained SOM to file for later use.

**Parameters**: `path` - File path (recommend .pkl or .npz extension).

### Method: load

```python
@classmethod
def load(cls, path: str) -> "SOMTrainer":
```

**Purpose**: Load a previously saved SOM from file.

---

# SOMPredictor Specification

## Class: SOMPredictor

**File**: `src/analitica/som/predictor.py`

**Purpose**: Inference module for making predictions with a trained SOM.

### Method: __init__

```python
def __init__(self, som: SOMTrainer) -> None:
```

**Parameters**:
- `som`: Trained SOMTrainer instance

**Docstring Example**:

```python
"""
SOM Predictor for inference on trained Self-Organizing Maps.

This class provides methods for finding Best Matching Units,
calculating quantization errors, and retrieving node data.

Example:
    >>> from analitica.som import SOMTrainer, SOMPredictor
    >>> 
    >>> # Train SOM
    >>> data = np.random.rand(100, 4)
    >>> som = SOMTrainer(x=10, y=10, input_len=4)
    >>> som.fit(data, epochs=5000)
    >>> 
    >>> # Create predictor
    >>> predictor = SOMPredictor(som)
    >>> 
    >>> # Find BMU for a single sample
    >>> bmu = predictor.bmu(data[0])
    >>> print(f"BMU at ({bmu[0]}, {bmu[1]})")
"""
```

### Method: bmu

```python
def bmu(self, x: np.ndarray) -> Tuple[int, int]:
```

**Purpose**: Find the Best Matching Unit (winning neuron) for a single input sample.

**Mathematical Definition**:
```
BMU(x) = argmin_{i,j} ||x - w_ij||₂
```

The Euclidean distance is computed between the input vector x and every neuron's weight vector. The neuron with minimum distance is returned as the BMU.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | np.ndarray | Input vector of shape (input_len,) or (1, input_len) |

**Returns**: Tuple[int, int] - (x_coord, y_coord) of the BMU

**Docstring Example**:

```python
"""
Find Best Matching Unit for a single input sample.

The BMU is the neuron whose weight vector has the smallest Euclidean
distance to the input vector. This is the fundamental operation for
mapping new data to the trained SOM.

Mathematical Definition:
    Given input vector x ∈ ℝ^d and neuron weights w_ij ∈ ℝ^d:
    
        BMU(x) = argmin_{i,j} ||x - w_ij||₂
    
    where ||·||₂ denotes the L2 (Euclidean) norm.

Parameters:
    x: np.ndarray of shape (input_len,) or (1, input_len)
        Single input vector. Must be normalized to [0, 1].

Returns:
    Tuple[int, int]: (x_coord, y_coord)
        Coordinates of the winning neuron on the SOM grid.
        x_coord ∈ [0, som.x), y_coord ∈ [0, som.y)

Example:
    >>> predictor = SOMPredictor(trained_som)
    >>> sample = np.array([0.5, 0.3, 0.7, 0.2])
    >>> x, y = predictor.bmu(sample)
    >>> print(f"Sample maps to neuron at ({x}, {y})")
    Sample maps to neuron at (4, 7)
"""
```

### Method: quantization_error

```python
def quantization_error(self, x: np.ndarray) -> float:
```

**Purpose**: Calculate the quantization error for a single sample - the distance from the sample to its BMU.

**Mathematical Definition**:
```
QE(x) = ||x - w_BMU(x)||
```

This measures how well the SOM represents this particular sample. A lower QE indicates the sample is well-represented by its BMU.

**Parameters**:
- `x`: Input vector of shape (input_len,)

**Returns**: float - The quantization error (distance to BMU)

**Docstring Example**:

```python
"""
Calculate quantization error for a sample.

The quantization error measures how well the SOM represents a given input.
It is the Euclidean distance between the input vector and its Best Matching
Unit's weight vector.

Mathematical Definition:
    QE(x) = ||x - w_BMU(x)||₂
    
Interpretation:
    - QE ≈ 0: Sample perfectly represented by a neuron
    - QE > 0.5: Sample is poorly represented (consider larger grid)
    - Very high QE: Sample may be an outlier

Parameters:
    x: np.ndarray of shape (input_len,)
        Input vector (normalized to [0, 1])

Returns:
    float: Quantization error (distance to BMU)

Example:
    >>> predictor = SOMPredictor(trained_som)
    >>> sample = np.array([0.5, 0.3, 0.7])
    >>> qe = predictor.quantization_error(sample)
    >>> print(f"QE = {qe:.4f}")
    QE = 0.1234
"""
```

### Method: get_node_data

```python
def get_node_data(self, node: Tuple[int, int]) -> np.ndarray:
```

**Purpose**: Retrieve all training samples that were assigned to a specific SOM node.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `node` | Tuple[int, int] | Node coordinates (x, y) |

**Returns**: np.ndarray of shape (n_samples_at_node, input_len) - the data points assigned to this node.

**Docstring Example**:

```python
"""
Get all samples assigned to a specific SOM node.

After training, each training sample is assigned to its BMU.
This method retrieves all samples that mapped to a particular node,
useful for analyzing cluster composition.

Parameters:
    node: Tuple[int, int]
        Node coordinates (x, y) where x ∈ [0, som.x), y ∈ [0, som.y)

Returns:
    np.ndarray of shape (n_samples, input_len)
        All training samples assigned to this node.
        Returns empty array if no samples assigned.

Example:
    >>> predictor = SOMPredictor(trained_som)
    >>> 
    >>> # Get all samples that mapped to node (5, 5)
    >>> node_samples = predictor.get_node_data((5, 5))
    >>> print(f"Node (5,5) has {len(node_samples)} samples")
    Node (5,5) has 12 samples
    
    >>> # Analyze these samples
    >>> print(f"Mean values: {node_samples.mean(axis=0)}")
"""
```

### Method: predict

```python
def predict(self, x: np.ndarray) -> Tuple[int, int, float]:
```

**Purpose**: Complete prediction for a single sample - returns BMU coordinates and distance.

**Parameters**:
- `x`: Input vector of shape (input_len,)

**Returns**: Tuple[int, int, float] - (node_x, node_y, distance)

**Docstring Example**:

```python
"""
Predict SOM coordinates and distance for a sample.

Convenience method that combines bmu() and quantization_error() 
in a single call, returning both the Best Matching Unit and
the distance to it.

Parameters:
    x: np.ndarray of shape (input_len,)
        Input vector (normalized to [0, 1])

Returns:
    Tuple[int, int, float]: (node_x, node_y, distance)
        - node_x, node_y: BMU coordinates on the grid
        - distance: Euclidean distance to BMU weight vector

Example:
    >>> predictor = SOMPredictor(trained_som)
    >>> sample = np.array([0.5, 0.3, 0.7, 0.2])
    >>> node_x, node_y, dist = predictor.predict(sample)
    >>> print(f"Maps to ({node_x}, {node_y}) with distance {dist:.4f}")
    Maps to (4, 7) with distance 0.1234
"""
```

---

# SOMVisualizer Specification

## Class: SOMVisualizer

**File**: `src/analitica/som/visualizer.py`

**Purpose**: Visualization utilities for Self-Organizing Maps including U-Matrix, component planes, and BMU plots.

### Method: __init__

```python
def __init__(self, som: SOMTrainer) -> None:
```

**Docstring Example**:

```python
"""
SOM Visualization utilities.

Provides methods for visualizing Self-Organizing Maps including:
    - U-Matrix: Shows distances between neighboring neurons
    - Component Planes: Feature-wise activation patterns
    - BMU Highlighting: Show where samples map on the grid
    - Training History: Quantization error over iterations

Example:
    >>> from analitica.som import SOMTrainer, SOMVisualizer
    >>> 
    >>> # Train SOM
    >>> data = np.random.rand(200, 5)
    >>> som = SOMTrainer(x=10, y=10, input_len=5)
    >>> som.fit(data, epochs=5000)
    >>> 
    >>> # Create visualizer
    >>> viz = SOMVisualizer(som)
    >>> 
    >>> # Plot U-Matrix
    >>> fig = viz.plot_umatrix()
    >>> viz.save_figure(fig, 'umatrix.png')
"""
```

### Method: plot_umatrix

```python
def plot_umatrix(self, cmap: str = "viridis", show: bool = True) -> matplotlib.figure.Figure:
```

**Purpose**: Generate U-Matrix (Unified Distance Matrix) visualization.

**Mathematical Definition**:

The U-Matrix shows the average distance between each neuron's weight vector and its neighbors:

```
U(i,j) = (1/8) × Σ_{neighbor} ||w_i,j - w_neighbor||
```

Where the sum is over all 8 neighboring neurons (or fewer at edges/corners).

**Interpretation**:
- **Low values (blue)**: Cluster centers - neurons with similar weights
- **High values (red/yellow)**: Cluster boundaries - dissimilar regions
- **Dark regions**: Dense clusters
- **Light regions**: Sparse regions / outliers

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cmap` | str | "viridis" | Matplotlib colormap |
| `show` | bool | True | If True, display the plot |

**Returns**: matplotlib.figure.Figure object

**Docstring Example**:

```python
"""
Plot U-Matrix (Unified Distance Matrix).

The U-Matrix is the primary tool for cluster visualization in SOM.
It shows the average Euclidean distance between each neuron and its
8 neighboring neurons.

Mathematical Definition:
    For neuron at position (i, j):
        U(i,j) = (1/N) × Σ ||w_ij - w_neighbor||
    
    where the sum is over all valid neighbors and N is the count.

Reading the U-Matrix:
    - Dark/Blue regions: Cluster centers (similar neurons)
    - Light/Yellow regions: Cluster boundaries (dissimilar)
    - Valleys: Natural clusters
    - Ridges: Cluster separation

Parameters:
    cmap: str, default "viridis"
        Matplotlib colormap. Options: 'viridis', 'plasma', 
        'inferno', 'magma', 'coolwarm', 'RdBu'
    show: bool, default True
        If True, calls plt.show()

Returns:
    matplotlib.figure.Figure: The generated figure

Example:
    >>> viz = SOMVisualizer(trained_som)
    >>> fig = viz.plot_umatrix(cmap='plasma', show=False)
    >>> viz.save_figure(fig, 'umatrix.png')
    
    # The plot shows 10×10 grid with color-coded distances
"""
```

### Method: plot_component_planes

```python
def plot_component_planes(
    self, 
    feature_names: Optional[List[str]] = None,
    cmap: str = "viridis",
    figsize: Optional[Tuple[int, int]] = None
) -> matplotlib.figure.Figure:
```

**Purpose**: Visualize how each input feature is distributed across the SOM grid.

**Mathematical Description**:

Each subplot shows one component (feature) of the weight vectors. For feature k:
```
Plane_k(i,j) = w_ij[k]
```

This reveals how each feature varies across the map, showing correlations between features and identifying regions where specific features have high or low values.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_names` | Optional[List[str]] | None | Names for each feature |
| `cmap` | str | "viridis" | Colormap for heatmaps |
| `figsize` | Optional[Tuple[int, int]] | None | Figure size |

**Returns**: matplotlib.figure.Figure with subplots for each feature

**Docstring Example**:

```python
"""
Plot component planes for all features.

Component planes show how each input feature is distributed across
the SOM grid. This is essential for understanding what each region
of the map represents in terms of the original features.

Mathematical Definition:
    For feature k at neuron (i, j):
        Plane_k(i,j) = w_ij[k]
    
    where w_ij is the weight vector of neuron at (i, j).

Interpretation:
    - Similar planes: Features are correlated
    - Opposite planes: Features are negatively correlated
    - Smooth gradients: Feature has continuous distribution
    - Patchy patterns: Feature has discrete categories

Parameters:
    feature_names: List[str], optional
        Names for each feature. If None, uses ["Feature 0", "Feature 1", ...]
    cmap: str, default "viridis"
        Matplotlib colormap
    figsize: Tuple[int, int], optional
        Figure size in inches. Default: (columns × 4, rows × 4)

Returns:
    matplotlib.figure.Figure: Grid of heatmaps, one per feature

Example:
    >>> # With named features
    >>> viz = SOMVisualizer(trained_som)
    >>> fig = viz.plot_component_planes(
    ...     feature_names=['age', 'income', 'spending', 'tenure'],
    ...     cmap='coolwarm'
    ... )
    >>> viz.save_figure(fig, 'component_planes.png')
    
    # Shows 4 heatmaps in a 2×2 grid
"""
```

### Method: plot_bmu

```python
def plot_bmu(
    self, 
    samples: np.ndarray,
    labels: Optional[List[str]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    marker: str = "o",
    markersize: int = 50,
    color_map: str = "tab10"
) -> matplotlib.axes.Axes:
```

**Purpose**: Overlay sample points on the SOM grid showing their Best Matching Units.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `samples` | np.ndarray | Required | Samples to plot, shape (n, input_len) |
| `labels` | Optional[List[str]] | None | Labels for each sample |
| `ax` | Optional[matplotlib.axes.Axes] | None | Existing axes to plot on |
| `marker` | str | "o" | Matplotlib marker style |
| `markersize` | int | 50 | Marker size |
| `color_map` | str | "tab10" | Colormap for labels |

**Returns**: matplotlib.axes.Axes with the plot

**Docstring Example**:

```python
"""
Plot Best Matching Units for samples on the SOM grid.

This visualization overlays input samples on the trained SOM,
showing which neurons they map to. Useful for:
    - Visualizing data distribution on the map
    - Comparing different groups/clusters
    - Tracking temporal data (time-series trajectories)

Parameters:
    samples: np.ndarray of shape (n_samples, input_len)
        Input samples to plot. Each sample's BMU is computed
        and marked on the grid.
    labels: List[str], optional
        Labels for each sample. If provided, color-coded by label.
    ax: matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    marker: str, default "o"
        Matplotlib marker style. Options: 'o', 's', '^', 'D', '*'
    markersize: int, default 50
        Size of markers
    color_map: str, default "tab10"
        Colormap name for label coloring

Returns:
    matplotlib.axes.Axes: Axes with the BMU markers

Example:
    >>> viz = SOMVisualizer(trained_som)
    >>> 
    >>> # Plot samples from different classes
    >>> class_a = data[:50]
    >>> class_b = data[50:]
    >>> 
    >>> ax = viz.plot_bmu(class_a, labels=['A']*50, color='blue')
    >>> viz.plot_bmu(class_b, ax=ax, color='red')
    >>> plt.title("Class Distribution on SOM")
"""
```

### Method: plot_training_history

```python
def plot_training_history(
    self, 
    figsize: Tuple[int, int] = (10, 5),
    log_scale: bool = False
) -> matplotlib.figure.Figure:
```

**Purpose**: Plot quantization error over training iterations to assess convergence.

**Mathematical Description**:

Shows how the average quantization error evolved during training:
```
QE(t) = (1/N) × Σ ||x_i - w_BMU(x_i)|| at iteration t
```

A good training should show:
- Sharp initial drop (ordering phase)
- Gradual decline and plateau (tuning phase)
- Final QE < initial QE × 0.1 (for well-trained SOM)

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `figsize` | Tuple[int, int] | (10, 5) | Figure dimensions |
| `log_scale` | bool | False | Use logarithmic y-axis |

**Returns**: matplotlib.figure.Figure

**Docstring Example**:

```python
"""
Plot training history (quantization error over epochs).

Shows how the quantization error evolved during training,
useful for diagnosing:
    - Convergence: Error should plateau
    - Underfitting: Error still decreasing at end
    - Overfitting: Error increases after minimum
    - Stable training: Smooth decline

Mathematical Definition:
    QE(t) = (1/N) × Σ ||x_i(t) - w_BMU(x_i, t)||₂
    
    where x_i are training samples and w_BMU are weights at epoch t.

Typical Training Curve:
    1. Epochs 0-20%: Sharp drop (ordering phase)
    2. Epochs 20-100%: Gradual decline (tuning phase)
    3. End: Plateau at minimum

Parameters:
    figsize: Tuple[int, int], default (10, 5)
        Figure size in inches
    log_scale: bool, default False
        If True, use logarithmic scale for y-axis

Returns:
    matplotlib.figure.Figure: Plot of QE vs epoch

Example:
    >>> viz = SOMVisualizer(trained_som)
    >>> fig = viz.plot_training_history()
    >>> viz.save_figure(fig, 'training_history.png')
    
    # Shows characteristic "L" shaped curve
"""
```

### Method: save_figure

```python
def save_figure(
    fig: matplotlib.figure.Figure, 
    path: str, 
    dpi: int = 150,
    bbox_inches: str = "tight"
) -> None:
```

**Purpose**: Export matplotlib figure to file.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fig` | matplotlib.figure.Figure | Required | Figure to save |
| `path` | str | Required | Output file path |
| `dpi` | int | 150 | Resolution |
| `bbox_inches` | str | "tight" | Bounding box mode |

**Docstring Example**:

```python
"""
Save matplotlib figure to file.

Utility method for exporting SOM visualizations with
sensible defaults for quality and file size.

Parameters:
    fig: matplotlib.figure.Figure
        The figure to save
    path: str
        Output file path. Format determined by extension:
        - .png: Portable Network Graphics
        - .pdf: Portable Document Format
        - .svg: Scalable Vector Graphics
        - .jpg: JPEG (lossy)
    dpi: int, default 150
        Resolution in dots per inch. Use 300 for print quality.
    bbox_inches: str, default "tight"
        Bounding box mode. "tight" removes excess whitespace.

Example:
    >>> viz = SOMVisualizer(som)
    >>> fig = viz.plot_umatrix(show=False)
    >>> viz.save_figure(fig, 'umatrix.png', dpi=300)
    >>> viz.save_figure(fig, 'umatrix.pdf')  # Vector format
"""
```

---

# SOMAnalyzer Specification

## Class: SOMAnalyzer

**File**: `src/analitica/som/analyzer.py`

**Purpose**: Quality metrics and analysis tools for Self-Organizing Maps.

### Method: __init__

```python
def __init__(self, som: SOMTrainer, data: Optional[np.ndarray] = None) -> None:
```

**Docstring Example**:

```python
"""
SOM Quality Analysis and Metrics.

Provides quantitative measures for assessing SOM quality:
    - Quantization Error: How well data fits the map
    - Topographic Error: How well topology is preserved
    - Node Distribution: How evenly data maps to neurons
    - Cluster Boundaries: Detection of cluster regions

Example:
    >>> from analitica.som import SOMTrainer, SOMAnalyzer
    >>> 
    >>> # Train SOM
    >>> data = np.random.rand(500, 4)
    >>> som = SOMTrainer(x=10, y=10, input_len=4)
    >>> som.fit(data, epochs=5000)
    >>> 
    >>> # Analyze quality
    >>> analyzer = SOMAnalyzer(som, data)
    >>> 
    >>> qe = analyzer.quantization_error()
    >>> te = analyzer.topographic_error()
    >>> print(f"QE: {qe:.4f}, TE: {te:.4f}")
"""
```

### Method: quantization_error

```python
def quantization_error(self, data: Optional[np.ndarray] = None) -> float:
```

**Purpose**: Calculate average quantization error for all data.

**Mathematical Definition**:
```
QE = (1/N) × Σ_{i=1}^{N} ||x_i - w_BMU(x_i)||₂
```

Where N is the number of samples and w_BMU(x_i) is the weight vector of the Best Matching Unit for sample x_i.

**Interpretation**:
- QE < 0.1: Excellent representation
- QE 0.1-0.2: Good representation
- QE 0.2-0.5: Acceptable representation
- QE > 0.5: Poor representation (consider larger grid)

**Parameters**:
- `data`: Optional np.ndarray - If provided, calculates QE for this data. If None, uses data from training.

**Returns**: float - Average quantization error

**Docstring Example**:

```python
"""
Calculate average quantization error.

The quantization error measures how well the SOM represents the data.
It is the average Euclidean distance between each sample and its
Best Matching Unit's weight vector.

Mathematical Definition:
    QE = (1/N) × Σ ||x_i - w_BMU(x_i)||₂
    
    where:
        N = number of samples
        x_i = input sample
        w_BMU(x_i) = weight vector of the Best Matching Unit

Interpretation Guidelines:
    - QE < 0.1: Excellent - data fits the map well
    - QE 0.1-0.2: Good - typical for well-tuned SOM
    - QE 0.2-0.5: Acceptable - may need more neurons
    - QE > 0.5: Poor - increase grid size or retrain

Parameters:
    data: np.ndarray, optional
        Data to evaluate. If None, uses training data.

Returns:
    float: Average quantization error

Example:
    >>> analyzer = SOMAnalyzer(trained_som, test_data)
    >>> qe = analyzer.quantization_error()
    >>> print(f"Quantization Error: {qe:.4f}")
    Quantization Error: 0.1523
    
    >>> # Compare train vs test error
    >>> train_qe = analyzer.quantization_error(train_data)
    >>> test_qe = analyzer.quantization_error(test_data)
    >>> print(f"Train QE: {train_qe:.4f}, Test QE: {test_qe:.4f}")
"""
```

### Method: topographic_error

```python
def topographic_error(self, data: Optional[np.ndarray] = None) -> float:
```

**Purpose**: Calculate topographic error - proportion of samples where first and second BMUs are not adjacent.

**Mathematical Definition**:
```
TE = (1/N) × count({i : BMU_1(x_i) and BMU_2(x_i) are not neighbors})
```

Two neurons are neighbors if they are adjacent horizontally, vertically, or diagonally on the grid.

**Interpretation**:
- TE < 0.05: Excellent topology preservation
- TE 0.05-0.1: Good topology
- TE 0.1-0.2: Acceptable
- TE > 0.2: Poor topology (increase sigma, more training)

**Parameters**:
- `data`: Optional np.ndarray - Data to evaluate

**Returns**: float - Topographic error (proportion)

**Docstring Example**:

```python
"""
Calculate topographic error.

Topographic error measures how well the SOM preserves topological
relationships. It calculates the proportion of samples where the
first and second Best Matching Units are not adjacent on the grid.

A well-trained SOM should have low topographic error, meaning that
similar inputs map to neurons that are close together.

Mathematical Definition:
    TE = (1/N) × Σ I(BMU_1(x_i) and BMU_2(x_i) are not neighbors)
    
    where:
        N = number of samples
        BMU_1(x) = first Best Matching Unit (closest)
        BMU_2(x) = second Best Matching Unit (2nd closest)
        I(condition) = 1 if true, 0 if false
        
    Two neurons are neighbors if their Manhattan distance ≤ 1:
        |x1 - x2| + |y1 - y2| ≤ 1

Interpretation Guidelines:
    - TE < 0.05: Excellent - topology well preserved
    - TE 0.05-0.10: Good - typical for trained SOM
    - TE 0.10-0.20: Acceptable - may need tuning
    - TE > 0.20: Poor - increase sigma or training epochs

Parameters:
    data: np.ndarray, optional
        Data to evaluate. If None, uses training data.

Returns:
    float: Topographic error (0 to 1, lower is better)

Example:
    >>> analyzer = SOMAnalyzer(trained_som, test_data)
    >>> te = analyzer.topographic_error()
    >>> print(f"Topographic Error: {te:.4f}")
    Topographic Error: 0.0312
    
    # Combined with QE for quality assessment
    >>> qe = analyzer.quantization_error()
    >>> print(f"Quality: QE={qe:.3f}, TE={te:.3f}")
"""
```

### Method: node_distribution

```python
def node_distribution(self, data: Optional[np.ndarray] = None) -> np.ndarray:
```

**Purpose**: Calculate the number of samples assigned to each node.

**Returns**: np.ndarray of shape (x, y) with hit counts per node

**Docstring Example**:

```python
"""
Calculate node hit distribution.

Shows how many training samples are assigned to each neuron.
This reveals:
    - Dense regions: Many samples map to same neurons
    - Sparse regions: Few or no samples
    - Unused neurons: Zero hits (potential outliers)

Returns:
    np.ndarray of shape (som.x, som.y)
        Count of samples assigned to each node

Example:
    >>> analyzer = SOMAnalyzer(trained_som, data)
    >>> dist = analyzer.node_distribution()
    >>> 
    >>> # Statistics
    >>> print(f"Total nodes used: {np.sum(dist > 0)}")
    >>> print(f"Max hits per node: {np.max(dist)}")
    >>> print(f"Mean hits per node: {np.mean(dist):.1f}")
    >>> print(f"Empty nodes: {np.sum(dist == 0)}")
"""
```

### Method: cluster_boundaries

```python
def cluster_boundaries(
    self, 
    threshold: Optional[float] = None
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
```

**Purpose**: Detect cluster boundaries based on U-matrix values.

**Mathematical Description**:

The method identifies boundary regions by finding edges where adjacent neurons have high U-matrix values (indicating dissimilar weight vectors, i.e., cluster separation).

If `threshold` is None, it uses the mean U-matrix value + one standard deviation.

**Parameters**:
- `threshold`: Optional float - U-matrix threshold for boundary detection. If None, uses auto-calculation.

**Returns**: List of boundary edges as ((x1,y1), (x2,y2)) tuples

**Docstring Example**:

```python
"""
Detect cluster boundaries from U-Matrix.

Identifies regions where adjacent neurons have significantly different
weight vectors, indicating cluster boundaries. These are the "ridges"
in the U-Matrix visualization.

Mathematical Definition:
    1. Compute U-matrix: U(i,j) = avg distance to neighbors
    2. Threshold: boundary if U > threshold
    3. Extract edges between boundary and non-boundary cells

Parameters:
    threshold: float, optional
        U-matrix value above which a cell is considered a boundary.
        If None: threshold = mean(U) + std(U)

Returns:
    List[Tuple[Tuple[int, int], Tuple[int, int]]]
        List of boundary edges as ((x1,y1), (x2,y2)) tuples

Example:
    >>> analyzer = SOMAnalyzer(trained_som)
    >>> boundaries = analyzer.cluster_boundaries()
    >>> print(f"Found {len(boundaries)} boundary edges")
    
    >>> # Use custom threshold for more/fewer boundaries
    >>> strict_boundaries = analyzer.cluster_boundaries(threshold=0.5)
    >>> loose_boundaries = analyzer.cluster_boundaries(threshold=0.3)
"""
```

---

# Integration Requirements

## ETL Pipeline Integration

The SOM modules MUST integrate with the existing ETL pipeline as follows:

```python
# Required: SOMTransformer for pipeline integration
from analitica.som import SOMTrainer
from analitica.etl import Pipeline, Transformer

class SOMTransformer(Transformer):
    """SOM transformation stage for ETL pipeline."""
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        epochs: int = 5000,
        normalize: bool = True,
        **som_kwargs
    ):
        self.grid_size = grid_size
        self.epochs = epochs
        self.normalize = normalize
        self.som_kwargs = som_kwargs
        self.som_ = None
    
    def fit(self, data: pd.DataFrame) -> "SOMTransformer":
        # Implementation...
        pass
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implementation...
        pass
```

---

# Acceptance Criteria

## SOMTrainer

- [ ] `__init__` accepts all specified parameters with correct defaults
- [ ] `fit()` normalizes input data automatically
- [ ] `fit()` implements two training phases (ordering + tuning)
- [ ] `transform()` returns correct BMU coordinates
- [ ] `fit_transform()` combines fit and transform correctly
- [ ] Weights are stored after training for later predictions

## SOMPredictor

- [ ] `bmu()` returns correct (x, y) for any input
- [ ] `quantization_error()` calculates correct distance
- [ ] `get_node_data()` returns all samples for a node
- [ ] `predict()` returns (node_x, node_y, distance)

## SOMVisualizer

- [ ] `plot_umatrix()` produces correct U-matrix visualization
- [ ] `plot_component_planes()` shows all features
- [ ] `plot_bmu()` overlays samples correctly
- [ ] `plot_training_history()` shows QE over epochs
- [ ] `save_figure()` exports plots correctly

## SOMAnalyzer

- [ ] `quantization_error()` calculates average QE correctly
- [ ] `topographic_error()` calculates TE correctly
- [ ] `node_distribution()` returns hit counts
- [ ] `cluster_boundaries()` detects boundaries

## Documentation

- [ ] Every class has comprehensive docstrings with examples
- [ ] Every method explains mathematical basis
- [ ] Parameter tables include typical values
- [ ] Usage examples are runnable

---

# Dependencies

- **minisom** (>=2.0): Core SOM implementation
- **numpy**: Numerical operations
- **matplotlib**: Visualization
- **pandas**: Data handling
