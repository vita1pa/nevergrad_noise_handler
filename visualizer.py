import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


class OptimizationVisualizer:
    """Real-time visualization for CMA-ES optimization."""
    
    def __init__(self, title='Optimization Progress'):
        """Initialize the visualizer with four subplots."""
        plt.ion()  # Enable interactive mode
        
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle(title)
        
        # Top-left plot: trajectory in 2D PCA space
        self.line, = self.ax1.plot([], [], 'bo-', markersize=5, alpha=0.6)
        self.current_point, = self.ax1.plot([], [], 'ro', markersize=10, label='Current')
        self.ax1.set_xlabel('PC1')
        self.ax1.set_ylabel('PC2')
        self.ax1.set_title('Trajectory (2D PCA)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Top-right plot: scalarized fitness over iterations
        self.line_fitness, = self.ax2.plot([], [], 'g-', linewidth=2)
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Best Fitness (Scalarized)')
        self.ax2.set_title('Overall Convergence')
        self.ax2.grid(True, alpha=0.3)
        
        # Bottom-left plot: R2 over iterations
        self.line_r2, = self.ax3.plot([], [], 'b-', linewidth=2)
        self.ax3.set_xlabel('Iteration')
        self.ax3.set_ylabel('R² Score')
        self.ax3.set_title('R² Convergence')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_ylim(0, 1)
        
        # Bottom-right plot: Total Incremental Contribution over iterations
        self.line_mape, = self.ax4.plot([], [], 'r-', linewidth=2)
        self.ax4.set_xlabel('Iteration')
        self.ax4.set_ylabel('Total Contribution')
        self.ax4.set_title('Total Contribution Convergence')
        self.ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
        self.trajectory = []
        self.fitness_history = []
        self.r2_history = []
        self.mape_history = []
    
    def update(self, mean, best_fitness, r2=None, mape=None):
        """Update the visualization with new data.
        
        Args:
            mean: Current mean solution vector
            best_fitness: Current best fitness value (scalarized)
            r2: Current R2 score (optional)
            mape: Current total incremental contribution (optional, kept as 'mape' for backward compatibility)
        """
        self.trajectory.append(mean.copy())
        self.fitness_history.append(best_fitness)
        if r2 is not None:
            self.r2_history.append(r2)
        if mape is not None:
            self.mape_history.append(mape)
        
        # Update trajectory plot
        if len(self.trajectory) >= 2:
            pca = PCA(n_components=2)
            traj_2d = pca.fit_transform(self.trajectory)
            
            # Update plot limits
            margin = 0.5
            self.ax1.set_xlim(min(traj_2d[:,0]) - margin, max(traj_2d[:,0]) + margin)
            self.ax1.set_ylim(min(traj_2d[:,1]) - margin, max(traj_2d[:,1]) + margin)
            
            # Update trajectory line
            self.line.set_data(traj_2d[:, 0], traj_2d[:, 1])
            self.current_point.set_data([traj_2d[-1, 0]], [traj_2d[-1, 1]])
        
        # Update scalarized fitness plot
        self.line_fitness.set_data(range(len(self.fitness_history)), self.fitness_history)
        self.ax2.set_xlim(0, len(self.fitness_history))
        if len(self.fitness_history) > 1:
            margin = 0.05 * (max(self.fitness_history) - min(self.fitness_history))
            self.ax2.set_ylim(min(self.fitness_history) - margin, max(self.fitness_history) + margin)
        
        # Update R2 plot
        if len(self.r2_history) > 0:
            self.line_r2.set_data(range(len(self.r2_history)), self.r2_history)
            self.ax3.set_xlim(0, len(self.r2_history))
        
        # Update Total Contribution plot
        if len(self.mape_history) > 0:
            self.line_mape.set_data(range(len(self.mape_history)), self.mape_history)
            self.ax4.set_xlim(0, len(self.mape_history))
            if len(self.mape_history) > 1:
                margin = 0.05 * (max(self.mape_history) - min(self.mape_history))
                self.ax4.set_ylim(max(0, min(self.mape_history) - margin), max(self.mape_history) + margin)
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def finalize(self):
        """Finalize the plot and keep it displayed."""
        plt.ioff()
        plt.show()
