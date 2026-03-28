# Proposal: SOM (Self-Organizing Maps) Integration

## Intent

Implement Self-Organizing Maps (SOM) as the **core unsupervised learning algorithm** for the analitica project. SOM provides dimensionality reduction while preserving topological relationships, enabling clustering and visualization of high-dimensional data. This is essential for ETL pipelines that need to discover patterns in raw data without labeled training examples.

## Background: What is a Self-Organizing Map?

### Mathematical Foundation

A Self-Organizing Map (SOM), also known as a Kohonen map, is an unsupervised neural network that learns to map high-dimensional input data onto a lower-dimensional (typically 2D) grid while preserving **topological relationships**. 

**Key Properties:**
1. **Topology Preservation**: Similar input vectors activate neurons that are close together on the map
2. **Dimensionality Reduction**: Projects n-dimensional data onto a 2D grid
3. **Competitive Learning**: Only the "winning" neuron (Best Matching Unit) updates its weights

### The SOM Algorithm

Given an input vector **x** ∈ ℝ^d and a grid of neurons with weight vectors **w**_ij ∈ ℝ^d:

1. **Initialization**: Initialize weight vectors (random, PCA, or random samples)

2. **Best Matching Unit (BMU) Finding**:
   ```
   BMU(x) = argmin_ij ||x - w_ij||
   ```
   The neuron whose weight vector is closest to the input vector wins.

3. **Weight Update** (Competitive Learning):
   ```
   w_ij(t+1) = w_ij(t) + α(t) · h_ij(t) · (x - w_ij(t))
   ```
   Where:
   - α(t) = learning rate at iteration t (decays over time)
   - h_ij(t) = neighborhood function (Gaussian typically)
   - h_ij(t) = exp(-d_ij² / (2σ(t)²))
   - d_ij = distance from BMU to neuron (i,j)
   - σ(t) = neighborhood radius at iteration t (decays over time)

4. **Two Training Phases**:
   - **Ordering Phase**: Large learning rate, large neighborhood → rough topology
   - **Tuning Phase**: Small learning rate, small neighborhood → fine adjustment

### Why SOM in ETL?

In ETL (Extract-Transform-Load) pipelines, SOM enables:
- **Data Exploration**: Discover natural clusters without predefined labels
- **Anomaly Detection**: Outliers map to sparse regions or isolated neurons
- **Feature Visualization**: Component planes reveal feature distributions
- **Data Reduction**: Replace high-dimensional records with neuron coordinates

## Scope

### In Scope

#### 1. SOMTrainer - Training Module
**File**: `src/analitica/som/trainer.py`

- Grid initialization strategies:
  - `random`: Random weight vectors
  - `pca`: Initialize along first two principal components
  - `random_samples`: Pick random data points as initial weights
- Training methods:
  - `train_random()`: Sequential with random sample order
  - `train_batch()`: Sequential through all samples
  - `train_batch_offline()`: Batch update per iteration
  - `train_batch_offline_fast()`: Numba-accelerated batch training
- Training phase configuration:
  - Ordering phase (rough mapping)
  - Tuning phase (fine-tuning)
- Learning rate decay functions:
  - `asymptotic_decay`: η(t) = η₀ / (1 + t/(max_iter/2))
  - `inverse_decay_to_zero`: η(t) = η₀ · C / (C + t)
  - `linear_decay_to_zero`: η(t) = η₀ · (1 - t/max_iter)
- Neighborhood functions:
  - `gaussian`: Smooth bell-shaped (default, recommended)
  - `mexican_hat`: Mexican hat (inverted center)
  - `bubble`: Constant within radius
  - `triangle`: Linear falloff

#### 2. SOMPredictor - Inference Module
**File**: `src/analitica/som/predictor.py`

- **BMU Finding**: `winner(x)` returns (x_idx, y_idx) for input vector
- **Quantization**: `quantization(data)` returns codebook vectors
- **Quantization Error**: Average distance to BMU
- **Node Assignment**: Map new data to trained SOM neurons
- **Activation Response**: Count how often each neuron wins

#### 3. SOMVisualizer - Visualization Module
**File**: `src/analitica/som/visualizer.py`

- **U-Matrix (Unified Distance Matrix)**:
  - Shows average distance between each neuron and its neighbors
  - High values = cluster boundaries
  - Low values = cluster centers
- **Component Planes**:
  - Visualize how each input feature is distributed across the map
  - Reveals correlations between features
- **BMU Highlighting**:
  - Overlay new data points on the map
  - Show trajectory for time-series data
- **Cluster Boundaries**:
  - Detect and draw boundaries based on U-matrix thresholding
- **Heat Maps**: Feature-wise activation intensity

#### 4. SOMAnalyzer - Quality Metrics Module
**File**: `src/analitica/som/analyzer.py`

- **Quantization Error (QE)**:
  ```
  QE = (1/N) · Σ ||x_i - w_BMU(x_i)||
  ```
  Measures how well data fits the map. Lower = better fit.
  
- **Topographic Error (TE)**:
  ```
  TE = (1/N) · count(first_BMU != second_BMU)
  ```
  Measures topology preservation. Lower = better topology.
  
- **Distribution Analysis**:
  - Neuron hit counts
  - Uniformity metric
  - Sparse/dense region detection
  
- **Convergence Metrics**:
  - Track QE over training iterations
  - Detect plateau conditions

### Out of Scope

- Deep SOM variants (e.g., Deep Autoencoder SOM)
- Growing Grid SOM (dynamic grid sizing)
- Supervised SOM variants
- Time-series specific SOM (SOM-TD)
- Web UI / interactive visualization

## Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     analitica.som                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ SOMTrainer   │  │ SOMPredictor │  │ SOMVisualizer│       │
│  │              │  │              │  │              │       │
│  │ - init_*()  │  │ - winner()   │  │ - umatrix()  │       │
│  │ - train_*()  │  │ - quantize() │  │ - planes()   │       │
│  │ - config     │  │ - error()     │  │ - bmu_plot() │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                             │
│  ┌──────────────┐                                          │
│  │ SOMAnalyzer  │                                          │
│  │              │                                          │
│  │ - QE, TE     │                                          │
│  │ - analyze()  │                                          │
│  │ - report()   │                                          │
│  └──────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   MiniSom 2.x   │
                    │ (dependencies)   │
                    └─────────────────┘
```

### Integration with ETL Pipeline

SOM will integrate with the existing ETL pipeline as a transformation stage:

```python
# Example usage in ETL context
pipeline = Pipeline([
    Source(...),
    Normalizer(),           # Existing normalizer
    SOMTransformer(         # NEW: SOM clustering
        grid_size=(10, 10),
        input_len=5,
        training_cycles=5000
    ),
    Destination(...)
])
```

### Parameter Selection Guidelines

| Parameter | Recommended Value | When to Adjust |
|-----------|------------------|----------------|
| Grid Size | 5·√N (rule of thumb) | N = number of samples |
| sigma | 1.0 initially | Decrease for fine-tuning |
| learning_rate | 0.5 initially | Higher for rough phase |
| iterations | 10-50 × grid_size | More for complex data |

### Example: Customer Segmentation

Given customer data (age, income, spending_score, tenure, region_encoded):

1. **Train SOM**: 10×10 grid, 5000 iterations
2. **Analyze**: QE = 0.15, TE = 0.02 (good topology)
3. **Visualize**: U-matrix shows 4 distinct clusters
4. **Assign**: New customers mapped to neurons → segment labels

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/analitica/som/__init__.py` | Modified | Add exports for new modules |
| `src/analitica/som/trainer.py` | New | SOMTrainer class |
| `src/analitica/som/predictor.py` | New | SOMPredictor wrapper |
| `src/analitica/som/visualizer.py` | New | Visualization utilities |
| `src/analitica/som/analyzer.py` | New | Quality metrics |
| `src/analitica/etl/transformer.py` | Modified | Add SOMTransformer to Pipeline |
| `tests/test_som.py` | New | Unit tests |
| `examples/demo_som.py` | New | Usage examples |

## Dependencies

- **minisom** (>=2.0): Core SOM implementation
- **numpy**: Numerical operations
- **matplotlib**: Visualization (for SOMVisualizer)
- **pandas**: Data handling

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|-------------|
| MiniSom version incompatibility | Low | Pin minisom>=2.0 in requirements |
| Large dataset memory issues | Medium | Use batch training, recommend data sampling |
| Poor topology preservation | Medium | Monitor TE, adjust sigma/neighborhood |
| Visualization performance | Low | Lazy computation, caching |
| Uninitialized weights cause poor results | Low | Default to PCA init for normalized data |

## Rollback Plan

1. Remove `src/analitica/som/` directory contents
2. Revert `src/analitica/som/__init__.py` to original (only SOMTrainer)
3. Remove any SOM-related imports from ETL transformer
4. Uninstall minisom if not used elsewhere
5. No database migrations or data cleanup needed

## Success Criteria

- [ ] SOMTrainer can initialize and train SOM with all initialization methods
- [ ] SOMPredictor correctly finds BMUs for new data
- [ ] SOMVisualizer produces U-matrix, component planes
- [ ] SOMAnalyzer computes QE and TE accurately
- [ ] Integration with ETL Pipeline works end-to-end
- [ ] Unit tests cover core functionality
- [ ] Documentation includes mathematical explanations for students
- [ ] Example scripts demonstrate real ETL use cases
