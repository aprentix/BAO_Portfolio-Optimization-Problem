import os
import pandas as pd

def save_results(results_dir, filename, weights, sharpe_ratio, annual_return):
    """
    Save portfolio optimization results to a CSV file.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{filename}_results.csv")
    results_df = pd.DataFrame([
        {"Company": k, "Weight": v, "Percentage": v * 100}
        for k, v in weights.items()
    ])
    results_df["Sharpe Ratio"] = sharpe_ratio
    results_df["Annual Return"] = annual_return
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")


def save_fitness_history(results_dir, filename, fitness_history):
    """
    Save fitness evolution history to a CSV file.
    """
    if fitness_history:
        fitness_file = os.path.join(results_dir, f"{filename}_fitness.csv")
        pd.DataFrame({"Generation": range(len(fitness_history)), "Fitness": fitness_history}).to_csv(fitness_file, index=False)
        print(f"Fitness history saved to {fitness_file}")


def save_diversity_history(results_dir, filename, diversity_history):
    """
    Save diversity evolution history to a CSV file.
    """
    if diversity_history:
        diversity_file = os.path.join(results_dir, f"{filename}_diversity.csv")
        pd.DataFrame({"Generation": range(len(diversity_history)), "Diversity": diversity_history}).to_csv(diversity_file, index=False)
        print(f"Diversity history saved to {diversity_file}")