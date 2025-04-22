# Implementation of Genetic Algorithm (GA) for Portfolio Optimization

This code implements a genetic algorithm (GA) to optimize a portfolio of assets by maximizing the Sharpe Ratio, subject to a 10% per-asset weight cap and a 4.20% risk-free rate.

---

## Imports

```python
import numpy as np
from inspyred import ec, benchmarks
from random import Random
```

- numpy: Used for numerical operations (e.g., matrix multiplication, array manipulation).

- inspyred.ec: Evolutionary computation library for implementing GA components (selection, mutation, crossover).

- inspyred.benchmarks: Base class for defining optimization problems.

- Random: Generates random numbers for stochastic operations in the GA.

---

## PortfolioOptimization Class

Defines the portfolio optimization problem, including constraints and the fitness function (Sharpe Ratio).

### Constructor: __init__ (PortfolioOptimization)

```python
def __init__(self, num_assets, mean_returns, cov_matrix, risk_free_rate=0.042):
```

Parameters:

- num_assets: Number of assets in the portfolio.

- mean_returns: Array of mean historical returns for each asset.

- cov_matrix: Covariance matrix of asset returns.

- risk_free_rate: Risk-free rate (default: 4.20%).

Key Actions:

- Inherits from benchmarks.Benchmark.

- Sets maximize=True to indicate a maximization problem.

- Uses RealBounder to constrain weights between 0 and 1.

### Method: generator

```python
def generator(self, random, args):
```

Generates initial candidate portfolios (chromosomes).

1. Creates random weights for each asset.

2. Normalizes weights to sum to 1.

### Method: evaluator

```python
def evaluator(self, candidates, args):
```

Computes the Sharpe Ratio for each candidate portfolio

1. Validate Weights:

    - Rejects candidates with weights that do not sum to 1 or contain negative values (assigns -np.inf fitness).

2. Calculate Sharpe Ratio:

    - Portfolio Return: port_return = np.dot(weights, mean_returns)

    - Portfolio Volatility: port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)

    - Sharpe Ratio: (port_return - risk_free_rate) / port_volatility

---

## GAPortfolioOptimization Class

Configures and runs the genetic algorithm to solve the portfolio optimization problem.

### Constructor: __init__ (GAPortfolioOptimization)

```python
def __init__(self, num_assets, mean_returns, cov_matrix, **kwargs):
```

Parameters:

- Inherits num_assets, mean_returns, and cov_matrix from PortfolioOptimization.

- Optional GA parameters (e.g., pop_size, max_generations).

Key Attributes:

- problem: Instance of PortfolioOptimization.

- GA hyperparameters: Population size, termination criteria, mutation rate, etc.

### Method: history_observer

```python
def history_observer(self, population, num_generations, num_evaluations, args):
```

Tracks the best fitness (Sharpe Ratio) across generations.

Appends the best fitness of each generation to best_fitness_history.

### Class Method: portfolio_repair

```python
@classmethod
def portfolio_repair(cls, random, candidates, args):
```

Constraint-handling method for evolutionary algorithms that ensures portfolio candidates adhere to:

Hard-constraints:

1. Weights must sum to `1.0`.

2. No weight can be negative.

Soft-constraints:

1. An asset shall not exceed `10%` allocation (soft constraint).

### Method: run

```python
def run(self, seed=None):
```

Executes the GA to evolve optimal portfolios.

1. Initialize GA: Configures crossover (blend_crossover), mutation (Gaussian_mutation), and repair (portfolio_repair).

2. Evolve Population: Generates and evaluates candidates over multiple generations.

3. Return Results: Best portfolio weights and Sharpe Ratio.
