import os
import pandas as pd

def prepare_file_saving(algorithm_type, correlation_level, params, root_path="results"):
    """
    Prepare the results directory and base filename for saving files.

    Args:
        algorithm_type (str): The optimization algorithm type ('ga' or 'pso').
        correlation_level (str): The correlation level ('low', 'medium', 'high', or None).
        params (dict): The algorithm-specific parameters.
        root_path (str): The root directory for saving results.

    Returns:
        tuple: (results_dir, base_filename)
    """
    # Simplify correlation level
    correlation_str = {
        "low": "L",
        "medium": "M",
        "high": "H",
        None: "N"
    }.get(correlation_level, "N")

    # Simplify parameter string
    if algorithm_type == "ga":
        param_str = f"ps-{params['pop_size']}_mg-{params['max_generations']}_mr-{params['mutation_rate']}"
    elif algorithm_type == "pso":
        param_str = f"ss-{params['swarm_size']}_mi-{params['max_iterations']}_w-{params['w']}"
    else:
        raise ValueError("Invalid algorithm type. Choose 'ga' or 'pso'.")

    # Define the results directory and base filename
    results_dir = os.path.join(root_path, algorithm_type)
    base_filename = f"exp_{correlation_str}_{param_str}"

    return results_dir, base_filename


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