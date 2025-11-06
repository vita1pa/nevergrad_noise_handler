import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


class OptimizationVisualizer:
    """Real-time visualization for CMA-ES optimization."""
    
    def __init__(self, title='Optimization Progress'):
        """Initialize the visualizer with two subplots."""
        plt.ion()  # Enable interactive mode
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle(title)
        
        # Left plot: trajectory in 2D PCA space
        self.line, = self.ax1.plot([], [], 'bo-', markersize=5, alpha=0.6)
        self.current_point, = self.ax1.plot([], [], 'ro', markersize=10, label='Current')
        self.ax1.set_xlabel('PC1')
        self.ax1.set_ylabel('PC2')
        self.ax1.set_title('Trajectory (2D PCA)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Right plot: fitness over iterations
        self.line_fitness, = self.ax2.plot([], [], 'g-', linewidth=2)
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Best Fitness')
        self.ax2.set_title('Convergence')
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
        self.trajectory = []
        self.fitness_history = []
    
    def update(self, mean, best_fitness):
        """Update the visualization with new data.
        
        Args:
            mean: Current mean solution vector
            best_fitness: Current best fitness value
        """
        self.trajectory.append(mean.copy())
        self.fitness_history.append(best_fitness)
        
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
        
        # Update fitness plot
        self.line_fitness.set_data(range(len(self.fitness_history)), self.fitness_history)
        self.ax2.set_xlim(0, len(self.fitness_history))
        self.ax2.set_ylim(-0.4, -0.2)
        # if len(self.fitness_history) > 1:
        #     self.ax2.set_ylim(min(self.fitness_history) - 1, max(self.fitness_history) + 5)
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def finalize(self):
        """Finalize the plot and keep it displayed."""
        plt.ioff()
        plt.show()
