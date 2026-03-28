"""
SOM Visualizer - Visualization tools for Self-Organizing Maps.

This module provides visualization capabilities for SOM analysis,
including U-Matrix, component planes, and BMU highlighting.

U-Matrix (Unified Distance Matrix)
==================================

The U-Matrix is the most important visualization for SOM interpretation.

It shows the average distance between each neuron and its neighbors:
- Large distances (warm colors): Cluster boundaries
- Small distances (cool colors): Clusters

How to compute:
For each neuron i, calculate:
    U_i = (1/N) * Σ ||w_i - w_j||

Where:
- w_i: Weight vector of neuron i
- w_j: Weight vectors of neighbors of i
- N: Number of neighbors
- ||...||: Euclidean distance

Interpreting the U-Matrix:
- Dark blue regions: Neurons close together, similar data
- Light yellow/red regions: Neurons far apart, cluster boundaries
- Valleys: Clusters
- Hills: Cluster boundaries
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from .trainer import SOMTrainer
from .predictor import SOMPredictor
from .exceptions import NotTrainedError


class SOMVisualizer:
    """
    Visualize Self-Organizing Maps.

    Provides methods to create various visualizations that help
    understand the SOM structure and data distribution.

    Example:
        >>> trainer = SOMTrainer(x=10, y=10, input_len=4)
        >>> trainer.fit(data)
        >>>
        >>> visualizer = SOMVisualizer(trainer)
        >>>
        >>> # Plot U-Matrix
        >>> visualizer.plot_umatrix()
        >>>
        >>> # Plot component plane for first feature
        >>> visualizer.plot_component_planes()
        >>>
        >>> # Highlight BMU for a sample
        >>> visualizer.plot_bmu(sample)
    """

    def __init__(self, trainer: SOMTrainer):
        """
        Initialize the visualizer.

        Args:
            trainer: A trained SOMTrainer instance.

        Raises:
            NotTrainedError: If the trainer hasn't been trained.
        """
        if not trainer.is_trained:
            raise NotTrainedError("Cannot create visualizer: SOM is not trained.")

        self._trainer = trainer
        self._som = trainer._som
        self._config = trainer.config
        self._weights = trainer.get_weights()

    def plot_umatrix(
        self,
        figsize: Tuple[int, int] = (10, 10),
        cmap: str = "viridis",
        show: bool = True,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the U-Matrix (Unified Distance Matrix).

        The U-Matrix shows distances between neighboring neurons:
        - High values (warm colors): Cluster boundaries
        - Low values (cool colors): Clusters

        This is the primary visualization for understanding
        the SOM structure.

        Args:
            figsize: Figure size as (width, height) in inches.
            cmap: Colormap for the heatmap.
                Recommended: 'viridis', 'coolwarm', 'YlOrRd'
            show: If True, display the plot immediately.
            title: Optional title for the plot.

        Returns:
            matplotlib Figure object.

        Raises:
            NotTrainedError: If SOM is not trained.

        Example:
            >>> fig = visualizer.plot_umatrix()
            >>> fig.savefig('umatrix.png')  # Save figure

            >>> # With custom appearance
            >>> fig = visualizer.plot_umatrix(
            ...     figsize=(12, 10),
            ...     cmap='coolwarm',
            ...     title='Customer Segmentation U-Matrix'
            ... )
        """
        # Get distance map from MiniSom
        distance_map = self._som.distance_map()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot U-Matrix
        im = ax.imshow(distance_map, cmap=cmap, origin="lower")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Distance")

        # Set labels
        ax.set_xlabel(f"Neuron X (0-{self._config.x - 1})")
        ax.set_ylabel(f"Neuron Y (0-{self._config.y - 1})")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(
                f"U-Matrix ({self._config.x}x{self._config.y} grid)\n"
                "High values = cluster boundaries, Low values = clusters"
            )

        # Add grid
        ax.set_xticks(np.arange(0, self._config.x, 1))
        ax.set_yticks(np.arange(0, self._config.y, 1))
        ax.grid(which="major", color="white", linestyle="-", linewidth=0.5)

        if show:
            plt.tight_layout()
            plt.show()

        return fig

    def plot_component_planes(
        self,
        feature_names: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: str = "RdYlBu_r",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot component planes for each feature.

        A component plane shows how a single feature is distributed
        across the SOM grid. Features with similar patterns across
        the map are correlated.

        Args:
            feature_names: List of feature names for labels.
                If None, uses 'Feature 0', 'Feature 1', etc.
            figsize: Figure size. If None, calculated from grid size.
            cmap: Colormap for heatmaps.
            show: If True, display the plot.

        Returns:
            matplotlib Figure object with subplots.

        Example:
            >>> # With feature names
            >>> fig = visualizer.plot_component_planes(
            ...     feature_names=['Age', 'Income', 'Score', 'Activity']
            ... )

            >>> # Save each component plane
            >>> for i, ax in enumerate(fig.axes):
            ...     ax.figure.savefig(f'component_{i}.png')
        """
        n_features = self._config.input_len

        # Determine grid layout
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle("Component Planes - Feature Distribution Across SOM")

        # Flatten axes for easy iteration
        if n_features == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i in range(n_features):
            ax = axes_flat[i]

            # Get weights for this feature
            component = self._weights[:, :, i]

            # Plot
            im = ax.imshow(component, cmap=cmap, origin="lower")
            plt.colorbar(im, ax=ax)

            # Labels
            feature_name = (
                feature_names[i]
                if feature_names and i < len(feature_names)
                else f"Feature {i}"
            )
            ax.set_title(feature_name)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        # Hide unused subplots
        for i in range(n_features, len(axes_flat)):
            axes_flat[i].axis("off")

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_bmu(
        self,
        sample: np.ndarray,
        data: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 10),
        cmap: str = "Blues",
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot the BMU for a sample on the SOM grid.

        Highlights where a specific sample maps to on the SOM,
        along with optionally showing nearby samples.

        Args:
            sample: Input sample vector.
            data: Optional full dataset to show sample distribution.
            figsize: Figure size.
            cmap: Colormap for sample density.
            show: If True, display the plot.

        Returns:
            matplotlib Figure object.

        Example:
            >>> # Highlight single sample
            >>> fig = visualizer.plot_bmu(sample)

            >>> # Show with data distribution
            >>> fig = visualizer.plot_bmu(sample, data=all_data)
        """
        predictor = SOMPredictor(self._trainer)
        bmu = predictor.bmu(sample)

        fig, ax = plt.subplots(figsize=figsize)

        # If data provided, show density
        if data is not None:
            if isinstance(data, pd.DataFrame):
                data = data.values

            # Get all BMUs for the data
            assignments = self._trainer.transform(data)

            # Count hits per neuron
            hits = np.zeros((self._config.y, self._config.x))
            for _, row in assignments.iterrows():
                hits[int(row["neuron_y"]), int(row["neuron_x"])] += 1

            # Plot density
            im = ax.imshow(hits, cmap=cmap, origin="lower", alpha=0.6)
            plt.colorbar(im, ax=ax, label="Sample Count")

        # Highlight BMU
        bmu_marker = plt.Rectangle(
            (bmu[0] - 0.5, bmu[1] - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3
        )
        ax.add_patch(bmu_marker)

        # Mark BMU center
        ax.plot(bmu[0], bmu[1], "r*", markersize=20, label=f"BMU ({bmu[0]}, {bmu[1]})")

        ax.set_xlim(-0.5, self._config.x - 0.5)
        ax.set_ylim(-0.5, self._config.y - 0.5)
        ax.set_xlabel(f"Neuron X (0-{self._config.x - 1})")
        ax.set_ylabel(f"Neuron Y (0-{self._config.y - 1})")
        ax.set_title(f"Best Matching Unit for Sample\nBMU: ({bmu[0]}, {bmu[1]})")
        ax.legend(loc="upper right")
        ax.set_aspect("equal")

        if show:
            plt.tight_layout()
            plt.show()

        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        path: str,
        dpi: int = 150,
    ) -> None:
        """
        Save a figure to file.

        Args:
            fig: matplotlib Figure to save.
            path: Output file path.
            dpi: Resolution in dots per inch.

        Example:
            >>> fig = visualizer.plot_umatrix(show=False)
            >>> visualizer.save_figure(fig, 'umatrix.png')
            >>> visualizer.save_figure(fig, 'umatrix.pdf')
        """
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
