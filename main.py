import cma
import numpy as np
import logging
from visualizer import OptimizationVisualizer  # Assuming this is provided; replace with matplotlib if needed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Toy data: 5 media channels X_data (n_samples x 5), sales Y
np.random.seed(42)
n_samples = 100
X_data = np.random.uniform(0, 10, (n_samples, 5))  # 5 channels spend
true_params = [0.7, 2.0, 5.0] * 5  # True [a,b,c] per channel (repeated for sim)

# Apply true transformations to generate Y
def adstock_transform(X_col, decay):
    adstock = np.zeros_like(X_col)
    adstock[0] = X_col[0]
    for t in range(1, len(X_col)):
        adstock[t] = X_col[t] + decay * adstock[t-1]
    return adstock

def hill_transform(adstock, slope, half):
    return adstock ** slope / (adstock ** slope + half ** slope)

true_transformed = np.zeros((n_samples, 5))
for i in range(5):
    a, b, c = true_params[i*3:(i+1)*3]
    adstock = adstock_transform(X_data[:, i], a)
    true_transformed[:, i] = hill_transform(adstock, b, c)
Y = 2.0 * np.sum(true_transformed, axis=1) + np.random.normal(0, 0.5, n_samples)  # Sum contributions + noise

# Multiobjective function: Params [a1,b1,c1, a2,b2,c2, ..., a5,b5,c5], transform each channel, fit linreg on stacked transformed, return [-R2, MAPE]
def multiobj_func(params, noise_level=0.1):
    params = np.asarray(params)  # Ensure numpy array
    transformed = np.zeros((n_samples, 5))
    for i in range(5):
        a, b, c = params[i*3:(i+1)*3]
        a = np.clip(a, 0, 1)  # Decay [0,1]
        b = np.clip(b, 0.1, 10)  # Slope [0.1,10]
        c = np.clip(c, 0.1, 20)  # Half [0.1,20]
        adstock = adstock_transform(X_data[:, i], a)
        transformed[:, i] = hill_transform(adstock, b, c)
    
    # Fit linreg: Y ~ sum(transformed channels); or stack for multi-feature
    reg = LinearRegression().fit(transformed, Y)
    Y_pred = reg.predict(transformed)
    
    r2 = r2_score(Y, Y_pred)
    mape = mean_absolute_percentage_error(Y, Y_pred)
    
    # Add noise
    noise_r2 = np.random.normal(0, noise_level * (abs(r2) + 1e-9))
    noise_mape = np.random.normal(0, noise_level * (mape + 1e-9))
    
    return [-r2 + noise_r2, mape + noise_mape]

# Logging callback (logs every gen)
def logging_callback(es):
    sigma = es.sigma
    loss = es.best.f  # Scalarized obj
    mean = es.mean
    logging.info(f"Iter {es.countevals}: sigma={sigma:.2e}, obj={loss:.2e}, mean={mean}")

# Setup pyCMA (dim=15 for 5*[a,b,c])
x0 = [0.5, 1.0, 5.0] * 5  # Initial [a,b,c] per channel
sigma0 = 0.5
# Bounds: [a,b,c] for each of 5 channels: a in [0,1], b in [0.1,10], c in [0.1,20]
lower_bounds = [0, 0.1, 0.1] * 5
upper_bounds = [1, 10, 20] * 5
options = {
    'popsize': 20,
    'verb_disp': 0,
    'verb_log': 0,
    'maxiter': 1000,
    'bounds': [lower_bounds, upper_bounds]
}

# NoiseHandler
noise_handler = cma.NoiseHandler(
    len(x0),
    reevals=4,
    maxevals=[1, 3, 10],
    aggregate=np.median,
)

# Optimization
es = cma.CMAEvolutionStrategy(x0, sigma0, options)

# Setup visualizer (assuming it's for trajectory; use matplotlib if not)
visualizer = OptimizationVisualizer(title='pyCMA Optimization on Multi-Channel AdHill')

iteration = 0
while not es.stop():
    X = es.ask()
    fits = []
    for x in X:
        reevals = int(noise_handler.evaluations)
        losses = [multiobj_func(x) for _ in range(reevals)]
        fit = np.median(losses, axis=0)  # Aggregate per objective using numpy directly
        fits.append(np.mean(fit))  # Scalarize mean for tell; handle Pareto separately if needed
    es.tell(X, fits)
    es.sigma *= noise_handler(X, fits, multiobj_func, es.ask)
    
    logging_callback(es)  # Explicit call if needed
    
    iteration += 1
    # Update visualization every 5 iterations
    if iteration % 5 == 0 or iteration == 1:
        # Get current best objectives (R2 and MAPE)
        best_objectives = multiobj_func(es.mean)  # [-R2, MAPE]
        r2_value = -best_objectives[0]  # Convert back to positive R2
        mape_value = best_objectives[1]
        visualizer.update(es.mean, es.best.f, r2=r2_value, mape=mape_value)

# Results
print(f"Best params: {es.result.xbest}")  # [a1,b1,c1,...,a5,b5,c5]
print(f"Best objectives: {multiobj_func(es.result.xbest)}")  # [-R2, MAPE]
print(f"Total evaluations: {es.countevals}")

# Finalize viz
visualizer.finalize()