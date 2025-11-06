import cma
import numpy as np
import pandas as pd
import logging
from visualizer import OptimizationVisualizer  # Assuming this is provided; replace with matplotlib if needed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load real data from MediaInvestment and Sales
print("Loading data from MediaInvestment.csv...")
data = pd.read_csv('MediaInvestment.csv')

# Define dependent variable (Y)
dep_var = 'Total_GMV'
Y = data[dep_var].values

# Define independent variables (X) - 8 media channels
indep_vars = ['TV', 'Digital', 'Sponsorship', 'Content Marketing', 
              'Online marketing', ' Affiliates', 'SEM', 'Radio']
X_data = data[indep_vars].values
X_data = np.nan_to_num(X_data, nan=0.0)  # Replace NaN with 0

n_samples = len(Y)
n_channels = len(indep_vars)

print(f"Loaded {n_samples} samples with {n_channels} media channels")
print(f"Y (Total_GMV) shape: {Y.shape}")
print(f"X (Media channels) shape: {X_data.shape}")
print(f"Channel names: {indep_vars}")

# Apply true transformations to generate Y
def adstock_transform(X_col, decay):
    adstock = np.zeros_like(X_col)
    adstock[0] = X_col[0]
    for t in range(1, len(X_col)):
        adstock[t] = X_col[t] + decay * adstock[t-1]
    return adstock

def hill_transform(adstock, slope, half):
    return adstock ** slope / (adstock ** slope + half ** slope)

# Multiobjective function: Params [a1,b1,c1, a2,b2,c2, ..., a8,b8,c8], transform each channel, fit linreg on stacked transformed, return [-R2, -Total_Contribution]
def multiobj_func(params, noise_level=0.1):
    params = np.asarray(params)  # Ensure numpy array
    transformed = np.zeros((n_samples, n_channels))
    for i in range(n_channels):
        a, b, c = params[i*3:(i+1)*3]
        a = np.clip(a, 0, 1)  # Decay [0,1]
        b = np.clip(b, 0.1, 100)  # Slope [0.1,100]
        c = np.clip(c, 0.1, 1)  # Half [0.1,1]
        adstock = adstock_transform(X_data[:, i], a)
        transformed[:, i] = hill_transform(adstock, b, c)
    
    # Fit linreg: Y ~ sum(transformed channels); or stack for multi-feature
    reg = LinearRegression().fit(transformed, Y)
    Y_pred = reg.predict(transformed)
    
    # Calculate individual channel contributions: beta * X_transformed
    betas = reg.coef_  # Coefficients for each channel
    contributions = transformed * betas  # Element-wise multiplication: (n_samples, n_channels)
    total_contributions = np.sum(contributions, axis=0)  # Sum over time for each channel
    
    # Total incremental contribution (sum of all channel contributions)
    total_incr_contribution = np.sum(total_contributions)
    
    r2 = r2_score(Y, Y_pred)
    
    # Add noise
    noise_r2 = np.random.normal(0, noise_level * (abs(r2) + 1e-9))
    noise_contrib = np.random.normal(0, noise_level * (abs(total_incr_contribution) + 1e-9))
    
    # Return negative values for minimization (we want to maximize both R2 and total contribution)
    return [-r2 + noise_r2, -total_incr_contribution + noise_contrib]

# Logging callback (logs every gen)
def logging_callback(es):
    sigma = es.sigma
    loss = es.best.f  # Scalarized obj
    mean = es.mean
    logging.info(f"Iter {es.countevals}: sigma={sigma:.2e}, obj={loss:.2e}, mean={mean}")

# Setup pyCMA (dim=24 for 8*[a,b,c])
x0 = [0.5, 50.0, 0.5] * n_channels  # Initial [a,b,c] per channel
sigma0 = 0.5
# Bounds: [a,b,c] for each of 8 channels: a in [0,1], b in [0.1,100], c in [0.1,1]
lower_bounds = [0, 0.1, 0.1] * n_channels
upper_bounds = [1, 100, 1] * n_channels
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
        # Get current best objectives (R2 and Total_Contribution)
        best_objectives = multiobj_func(es.mean)  # [-R2, -Total_Contrib]
        r2_value = -best_objectives[0]  # Convert back to positive R2
        total_contrib = -best_objectives[1]  # Convert back to positive total contribution
        visualizer.update(es.mean, es.best.f, r2=r2_value, mape=total_contrib)

# Results
print(f"Best params: {es.result.xbest}")  # [a1,b1,c1,...,a8,b8,c8]
best_objectives = multiobj_func(es.result.xbest)
print(f"Best R2: {-best_objectives[0]:.4f}")
print(f"Best Total Contribution: {-best_objectives[1]:.2e}")
print(f"Total evaluations: {es.countevals}")

# Calculate and display incremental contributions for each channel
print("\n" + "="*80)
print("INCREMENTAL CONTRIBUTIONS BY CHANNEL:")
print("="*80)

# Transform data with best parameters
transformed_best = np.zeros((n_samples, n_channels))
for i in range(n_channels):
    a, b, c = es.result.xbest[i*3:(i+1)*3]
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0.1, 100)
    c = np.clip(c, 0.1, 1)
    adstock = adstock_transform(X_data[:, i], a)
    transformed_best[:, i] = hill_transform(adstock, b, c)

# Fit model and get coefficients
reg_final = LinearRegression().fit(transformed_best, Y)
betas = reg_final.coef_

# Calculate contributions: beta * X_transformed
contributions = transformed_best * betas
total_contributions = np.sum(contributions, axis=0)
contribution_percentages = 100 * total_contributions / np.sum(total_contributions)

print("\nOptimized parameters and contributions per channel:")
print(f"{'Channel':<20} {'a (decay)':<12} {'b (slope)':<12} {'c (half-sat)':<12} {'Beta':<12} {'Contribution':<15} {'% of Total':<10}")
print("-" * 110)
for i in range(n_channels):
    a, b, c = es.result.xbest[i*3:(i+1)*3]
    print(f"{indep_vars[i]:<20} {a:<12.4f} {b:<12.4f} {c:<12.4f} {betas[i]:<12.4f} {total_contributions[i]:<15.2e} {contribution_percentages[i]:<10.2f}%")

print(f"\nTotal Contribution: {np.sum(total_contributions):.2e}")
print(f"Intercept: {reg_final.intercept_:.2e}")

# Finalize viz
visualizer.finalize()