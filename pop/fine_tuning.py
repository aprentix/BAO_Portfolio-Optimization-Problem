import time
import numpy as np
import pandas as pd
from itertools import product
from runner import runner

NUM_RUNS = 30
NUM_COMPANIES = 50
START_DATE = "2015-01-01"
END_DATE = "2020-01-01"
RISK_FREE_RATE = 0.042
DATASET = "dataset"

# Define GA hyperparameter grid
ga_param_grid = {
    "pop_size": [50, 100],
    "mutation_rate": [0.05, 0.1],
    "gaussian_stdev": [0.05, 0.1],
    "num_elites": [1, 2]
}

# Define PSO hyperparameter grid
pso_param_grid = {
    "pop_size": [50, 100],
    "w": [0.5, 0.7],
    "c1": [1.0, 1.5],
    "c2": [1.5, 2.0]
}


def get_param_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def evaluate_config(algorithm_type: str, config: dict):
    scores, times, returns = [], [], []
    for _ in range(NUM_RUNS):
        start = time.time()
        sharpe, ret, _ = runner(
            algorithm_type=algorithm_type,
            dataset_folder_name=DATASET,
            num_companies=NUM_COMPANIES,
            risk_free_rate_annual=RISK_FREE_RATE,
            start_date=START_DATE,
            end_date=END_DATE,
            **config
        )
        scores.append(sharpe)
        returns.append(ret)
        times.append(time.time() - start)
    return np.mean(scores), np.std(scores), np.mean(returns), np.mean(times), config


REPAIR_METHODS = ["normalize", "clip", "restart", "shrink"]

def main():
    results = []

    # GA tuning
    print("\nRunning GA tuning...")
    for repair in REPAIR_METHODS:
        for config in get_param_combinations(ga_param_grid):
            print(f"Testing GA config: {config} + repair: {repair}")
            mean_score, std_score, mean_return, mean_time, used_config = evaluate_config("ga", {
                **config,
                "repair_method": repair
            })
            results.append({
                "algorithm": "GA",
                "repair_method": repair,
                "mean_sharpe": mean_score,
                "std_sharpe": std_score,
                "mean_return": mean_return,
                "mean_time": mean_time,
                **used_config
            })

    # PSO tuning
    print("\nRunning PSO tuning...")
    for repair in REPAIR_METHODS:
        for config in get_param_combinations(pso_param_grid):
            print(f"Testing PSO config: {config} + repair: {repair}")
            mean_score, std_score, mean_return, mean_time, used_config = evaluate_config("pso", {
                **config,
                "repair_method": repair
            })
            results.append({
                "algorithm": "PSO",
                "repair_method": repair,
                "mean_sharpe": mean_score,
                "std_sharpe": std_score,
                "mean_return": mean_return,
                "mean_time": mean_time,
                **used_config
            })

    df = pd.DataFrame(results)
    df.to_csv("fine_tuning_results.csv", index=False)
    print("\nSaved fine-tuning results to fine_tuning_results.csv")

if __name__ == "__main__":
    main()
