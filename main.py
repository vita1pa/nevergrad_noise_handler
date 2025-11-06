import cma
import numpy as np
import logging
from visualizer import OptimizationVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Noisy Rastrigin (dim=5)
def noisy_rastrigin(x, noise_level=0.1):
    n = len(x)
    value = 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    noise = np.random.normal(0, noise_level * (np.sqrt(value) + 1e-9))
    return value + noise

# Logging callback (logs everything every tell, via options['callback'])
def logging_callback(es):
    sigma = es.sigma
    loss = es.best.f  # Last best loss
    mean = es.mean
    logging.info(f"Iter {es.countevals}: sigma={sigma:.2e}, loss={loss:.2e}, mean={mean}")

# Setup pyCMA (dim=5, popsize=20)
x0 = 5 * [0.0]
sigma0 = 0.5
options = {
    'popsize': 20,
    'verb_disp': 0,
    'verb_log': 0,
    'maxiter': 50,  # Budget / popsize ~1000 evals
}

# NoiseHandler
noise_handler = cma.NoiseHandler(
    len(x0),
    reevals=4,
    maxevals=[1, 3, 10],
    aggregate=np.median,
    # noise_change_sigma_exponent=0.8
)

# Optimization with ask/tell + NoiseHandler
es = cma.CMAEvolutionStrategy(x0, sigma0, options)

# Setup visualizer
visualizer = OptimizationVisualizer(title='pyCMA Optimization on Noisy Rastrigin')

iteration = 0
while not es.stop():
    X = es.ask()
    fits = []
    for x in X:
        reevals = int(noise_handler.evaluations)
        losses = [noisy_rastrigin(x) for _ in range(reevals)]
        fit = noise_handler.f_aggregate(losses)
        fits.append(fit)
    es.tell(X, fits)
    es.sigma *= noise_handler(X, fits, noisy_rastrigin, es.ask)
    
    logging_callback(es)  # Log every generation
    
    iteration += 1
    # Update visualization every 5 iterations
    if iteration % 5 == 0 or iteration == 1:
        visualizer.update(es.mean, es.best.f)

# Results
print(f"Best solution: {es.result.xbest}")
print(f"Best fitness: {es.result.fbest}")
print(f"Total evaluations: {es.countevals}")

# Keep the final plot displayed
visualizer.finalize()