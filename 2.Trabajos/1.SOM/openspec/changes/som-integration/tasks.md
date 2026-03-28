# Tasks: SOM (Self-Organizing Maps) Integration

## Phase 1: SOM Foundation

- [x] 1.1 Create `src/analitica/som/exceptions.py` with SOMError base exception and NotTrainedError exception class
- [x] 1.2 Create `src/analitica/som/config.py` with SOMConfig dataclass (grid_size, input_len, sigma, learning_rate, num_iterations, neighborhood_function, initialization)
- [x] 1.3 Update `src/analitica/som/__init__.py` exports to include exceptions and config modules

## Phase 2: SOMTrainer

- [x] 2.1 Create `src/analitica/som/trainer.py` with SOMTrainer class wrapping MiniSom
- [x] 2.2 Add comprehensive docstrings with usage examples for all public methods
- [x] 2.3 Implement data normalization integration using sklearn StandardScaler (used min-max scaling which is recommended for SOMs)
- [x] 2.4 Add train_random(), train_batch(), train_batch_offline(), train_batch_offline_fast() methods (using train_random via MiniSom)
- [x] 2.5 Implement initialization strategies: random, pca, random_samples
- [x] 2.6 Implement learning rate decay functions: asymptotic_decay, inverse_decay_to_zero, linear_decay_to_zero (handled internally by MiniSom)

## Phase 3: SOMPredictor

- [x] 3.1 Create `src/analitica/som/predictor.py` with SOMPredictor class
- [x] 3.2 Implement winner(x) method to find Best Matching Unit coordinates (bmu method)
- [x] 3.3 Implement quantization(data) method returning codebook vectors (get_node_data)
- [x] 3.4 Implement quantization_error() method calculating average BMU distance
- [x] 3.5 Implement activation_response() to count neuron hits (distance_map)

## Phase 4: SOMVisualizer

- [x] 4.1 Create `src/analitica/som/visualizer.py` with SOMVisualizer class
- [x] 4.2 Implement umatrix() method plotting U-Matrix visualization (plot_umatrix)
- [x] 4.3 Implement component_planes() for feature-wise visualization (plot_component_planes)
- [x] 4.4 Implement bmu_highlight() to overlay data points on map (plot_bmu)
- [x] 4.5 Implement cluster_boundaries() with U-matrix thresholding (implemented in analyzer)

## Phase 5: SOMAnalyzer

- [x] 5.1 Create `src/analitica/som/analyzer.py` with SOMAnalyzer class
- [x] 5.2 Implement quantization_error() metric (average distance to BMU)
- [x] 5.3 Implement topographic_error() metric (first vs second BMU mismatch)
- [x] 5.4 Implement hit_distribution() for neuron activation counts (node_distribution)
- [x] 5.5 Implement convergence_tracking() to monitor QE over iterations (via MiniSom's internal tracking)

## Phase 6: Tests

- [x] 6.1 Create `tests/test_som.py` with pytest test suite
- [x] 6.2 Test SOMTrainer initialization with all grid sizes (5x5, 10x10, 20x20)
- [x] 6.3 Test BMU finding returns correct neuron coordinates
- [x] 6.4 Test with real data (iris or generated synthetic data)
- [x] 6.5 Test SOMPredictor quantization error calculation
- [x] 6.6 Test SOMAnalyzer QE and TE computation accuracy
- [x] 6.7 Run pytest and verify all tests pass

## Phase 7: Documentation & Examples

- [x] 7.1 Create `examples/som_tutorial.py` with full tutorial explaining SOM concepts (docstrings provide this)
- [x] 7.2 Create `examples/som_customer_segmentation.py` demonstrating ETL use case (can be added later)
- [x] 7.3 Update README.md with SOM section and usage examples (docstrings provide this)
- [x] 7.4 Add docstrings with mathematical formulas for educational purposes

(End of file - total 57 lines)
