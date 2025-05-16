"""
Fine Tuning Utilities for Portfolio Optimization

This module provides simplified utilities for fine-tuning portfolio optimization algorithms.
It supports Genetic Algorithms (GA) and Particle Swarm Optimization (PSO) with parallel execution.
"""

import os
import pandas as pd
import numpy as np
from itertools import product
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import from pop module
from pop.runner import runner
from pop.util.file_saver import (
    prepare_file_saving, save_results, save_fitness_history, save_diversity_history
)


def get_results_path(filename, algorithm=None):
    """
    Returns the path to the results file, optionally within an algorithm subfolder.

    Args:
        filename (str): Name of the file
        algorithm (str, optional): Algorithm subfolder (ga, pso). Defaults to None.

    Returns:
        str: Full path to the results file
    """
    # Get project root (current working directory or parent containing experiments folder)
    current_dir = os.getcwd()
    results_dir = os.path.join(current_dir, "experiments", "results")

    if algorithm:
        return os.path.join(results_dir, algorithm, filename)
    return os.path.join(results_dir, filename)


def generate_base_filename(config, algorithm, correlation_level=None):
    """
    Generates a standardized filename based on algorithm configuration.

    Args:
        config (dict): Algorithm configuration parameters
        algorithm (str): Algorithm type (ga, pso)
        correlation_level (str, optional): Correlation level (low, medium, high)

    Returns:
        str: Formatted filename
    """
    correlation_str = {
        "low": "L",
        "medium": "M",
        "high": "H",
        None: "N"
    }.get(correlation_level, "N")

    if algorithm == "ga" or algorithm == "GA":
        pop_size = int(config["pop_size"]) if "pop_size" in config and not pd.isna(
            config["pop_size"]) else "NA"
        max_gen = int(config["max_generations"]) if "max_generations" in config and not pd.isna(
            config["max_generations"]) else "NA"
        mutation_rate = config["mutation_rate"] if "mutation_rate" in config else "NA"
        return f"exp_{correlation_str}_ps-{pop_size}_mg-{max_gen}_mr-{mutation_rate}"
    elif algorithm == "pso" or algorithm == "PSO":
        swarm_size = int(config["swarm_size"]) if "swarm_size" in config and not pd.isna(
            config["swarm_size"]) else "NA"
        max_iter = int(config["max_iterations"]) if "max_iterations" in config and not pd.isna(
            config["max_iterations"]) else "NA"
        w = config["w"] if "w" in config else "NA"
        return f"exp_{correlation_str}_ss-{swarm_size}_mi-{max_iter}_w-{w}"
    else:
        return "unknown_config"


def get_param_combinations(param_grid):
    """
    Generate all possible parameter combinations from a parameter grid.

    Args:
        param_grid (dict): Dictionary with parameter names as keys and lists of values

    Returns:
        list: List of dictionaries, each containing a unique parameter combination
    """
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def evaluate_config(algorithm_type, config, repair_method, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed):
    """
    Evaluate a single algorithm configuration across multiple runs.

    Args:
        algorithm_type (str): Type of algorithm ("ga" or "pso")
        config (dict): Algorithm configuration parameters
        repair_method (str): Method to repair invalid solutions
        num_runs (int): Number of runs to perform
        dataset (str): Path to dataset
        num_companies (int): Number of companies to include
        risk_free_rate (float): Annual risk-free rate
        start_date (str): Start date for analysis
        end_date (str): End date for analysis
        correlation_level (str): Correlation level
        seed (int): Random seed

    Returns:
        tuple: Performance metrics and configuration details
    """

    scores, times, returns, generations = [], [], [], []
    results_dir, base_filename = prepare_file_saving(
        algorithm_type=algorithm_type,
        correlation_level=correlation_level,
        params=config,
        root_path=os.path.join(os.getcwd(), "experiments", "results")
    )

    for run_id in range(1, num_runs + 1):
        start = time.time()
        try:
            results, fitness_history, diversity_history = runner(
                algorithm_type=algorithm_type,
                dataset_folder_name=dataset,
                num_companies=num_companies,
                risk_free_rate_annual=risk_free_rate,
                start_date=start_date,
                end_date=end_date,
                correlation_level=correlation_level,
                seed=seed + run_id,
                repair_method=repair_method,
                **config
            )
            sharpe_ratio, annual_return, weights = results
            runtime = time.time() - start
            num_generations = len(fitness_history)

            scores.append(sharpe_ratio)
            returns.append(annual_return)
            times.append(runtime)
            generations.append(num_generations)

            run_filename = f"{base_filename}_run{run_id}"
            save_results(results_dir, run_filename, weights,
                         sharpe_ratio, annual_return)
            save_fitness_history(results_dir, run_filename, fitness_history)
            save_diversity_history(
                results_dir, run_filename, diversity_history)

            print(
                f"✅ Run {run_id}/{num_runs} completed for {algorithm_type} - {config}")
        except Exception as e:
            print(
                f"Error during evaluation of {algorithm_type} with config {config}: {e}")

    # Calculate summary statistics
    mean_score = np.mean(scores) if scores else np.nan
    std_score = np.std(scores) if scores else np.nan
    mean_return = np.mean(returns) if returns else np.nan
    mean_time = np.mean(times) if times else np.nan
    mean_generations = np.mean(generations) if generations else np.nan

    # Save aggregated results
    if scores:
        aggregated_filename = f"{base_filename}_aggregated"
        save_results(results_dir, aggregated_filename,
                     weights, mean_score, mean_return)
        save_fitness_history(results_dir, aggregated_filename, fitness_history)
        save_diversity_history(
            results_dir, aggregated_filename, diversity_history)

    # Add the algorithm and repair method to the config for the return value
    config_with_meta = config.copy()
    config_with_meta["algorithm"] = algorithm_type.upper()
    config_with_meta["repair_method"] = repair_method

    return mean_score, std_score, mean_return, mean_time, mean_generations, config_with_meta


def fine_tune_algorithms_parallel(num_runs, dataset, num_companies, risk_free_rate, start_date, end_date,
                                  correlation_level, seed, ga_param_grid, pso_param_grid,
                                  repair_methods, max_workers=None):
    """
    Fine-tune GA and PSO algorithms in parallel across all parameter combinations.

    Args:
        num_runs (int): Number of runs per configuration
        dataset (str): Path to dataset
        num_companies (int): Number of companies to include
        risk_free_rate (float): Annual risk-free rate
        start_date (str): Start date for analysis
        end_date (str): End date for analysis
        correlation_level (str): Correlation level (low, medium, high)
        seed (int): Random seed
        ga_param_grid (dict): Grid of GA parameters to explore
        pso_param_grid (dict): Grid of PSO parameters to explore
        repair_methods (list): List of repair methods to test
        max_workers (int, optional): Maximum number of parallel workers

    Returns:
        list: Results of all configurations tested
    """
    # Create a list of all parameter combinations to test
    jobs = []
    for algo_type, param_grid in [("GA", ga_param_grid), ("PSO", pso_param_grid)]:
        for repair in repair_methods:
            for config in get_param_combinations(param_grid):
                jobs.append((algo_type.lower(), config, repair, num_runs, dataset, num_companies,
                            risk_free_rate, start_date, end_date, correlation_level, seed))

    # Run evaluations in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for job in jobs:
            # Unpack the job tuple to pass individual arguments
            algo_type, config, repair, runs, ds, companies, rate, start_date, end_date, corr_level, seed_val = job
            futures.append(executor.submit(
                evaluate_config,
                algo_type, config, repair, runs, ds, companies, rate, start_date, end_date, corr_level, seed_val
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fine-tuning configs"):
            try:
                mean_score, std_score, mean_return, mean_time, mean_generations, used_config = future.result()
                results.append({
                    "algorithm": used_config.get("algorithm", "Unknown"),
                    "repair_method": used_config.get("repair_method", "Unknown"),
                    "mean_sharpe": mean_score,
                    "std_sharpe": std_score,
                    "mean_return": mean_return,
                    "mean_time": mean_time,
                    "mean_generations": mean_generations,
                    **{k: v for k, v in used_config.items() if k not in ["algorithm", "repair_method"]}
                })
            except Exception as e:
                print(f"Error in parallel evaluation: {str(e)}")

    # Save results for each algorithm type
    for algo_type in ["GA", "PSO"]:
        algo_results = [r for r in results if r['algorithm'] == algo_type]
        if algo_results:
            df = pd.DataFrame(algo_results)
            result_path = get_results_path(
                "fine_tuning_results.csv", algo_type.lower())
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            df.to_csv(result_path, index=False)
            print(
                f"✅ Fine-tuning results for {algo_type} saved to '{result_path}'")

    return results


def select_configs(df, algorithm):
    """
    Select the best, median, and worst configurations based on Sharpe ratio.

    Args:
        df (DataFrame): DataFrame containing fine-tuning results
        algorithm (str): Algorithm type ("GA" or "PSO")

    Returns:
        tuple: Best, median, and worst configurations
    """
    df_algo = df[df["algorithm"] == algorithm].copy()

    if len(df_algo) == 0:
        raise ValueError(f"No configurations found for algorithm {algorithm}")

    # Sort the dataframe by mean_sharpe in descending order
    df_algo = df_algo.sort_values(
        by="mean_sharpe", ascending=False).reset_index(drop=True)

    # Select best, median, and worst configurations
    best_config = df_algo.iloc[0].copy() if len(df_algo) > 0 else None
    median_idx = len(df_algo) // 2
    median_config = df_algo.iloc[median_idx].copy() if len(
        df_algo) > median_idx else None
    worst_config = df_algo.iloc[-1].copy() if len(df_algo) > 0 else None

    # Add the quality attribute to each configuration
    if best_config is not None:
        best_config["quality"] = "best"
    if median_config is not None:
        median_config["quality"] = "median"
    if worst_config is not None:
        worst_config["quality"] = "worst"

    return best_config, median_config, worst_config


def run_single_config(config, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed):
    """
    Run a single configuration for multiple runs and collect results.

    Args:
        config (dict or Series): Configuration parameters
        num_runs (int): Number of runs to perform
        dataset (str): Path to dataset
        num_companies (int): Number of companies to include
        risk_free_rate (float): Annual risk-free rate
        start_date (str): Start date for analysis
        end_date (str): End date for analysis
        correlation_level (str): Correlation level
        seed (int): Random seed

    Returns:
        DataFrame: Results for this configuration
    """

    # Ensure config is a dictionary
    if hasattr(config, 'to_dict'):
        config = config.to_dict()
    else:
        config = dict(config)

    # Ensure repair_method is present
    if "repair_method" not in config or pd.isna(config["repair_method"]) or config["repair_method"] == "":
        config = config.copy()
        config["repair_method"] = "normalize"

    algo = config["algorithm"].lower()
    quality = config["quality"]

    # Prepare results collection
    results = []
    fitness_histories = []
    diversity_histories = []

    # Run the algorithm multiple times
    for run_id in range(1, num_runs + 1):
        try:
            # Prepare parameters for runner
            runner_params = {k: v for k, v in config.items()
                             if k not in ["algorithm", "quality", "repair_method", "mean_sharpe",
                                          "std_sharpe", "mean_return", "mean_time", "mean_generations"]}

            # Convert integer parameters from string or float to int
            int_params = ["pop_size", "num_elites",
                          "max_generations", "swarm_size", "max_iterations"]
            for param in int_params:
                if param in runner_params and not pd.isna(runner_params[param]):
                    runner_params[param] = int(float(runner_params[param]))

            # Run the algorithm
            start = time.time()
            run_output, fitness_history, diversity_history = runner(
                algorithm_type=algo,
                dataset_folder_name=dataset,
                num_companies=num_companies,
                risk_free_rate_annual=risk_free_rate,
                start_date=start_date,
                end_date=end_date,
                correlation_level=correlation_level,
                seed=seed + run_id,
                repair_method=config["repair_method"],
                **runner_params
            )

            # Extract results
            sharpe_ratio, annual_return, weights = run_output
            runtime = time.time() - start

            # Skip invalid results
            if not np.isfinite(sharpe_ratio) or not np.isfinite(annual_return):
                continue

            # Collect results
            results.append({
                "algorithm": config["algorithm"],
                "quality": quality,
                "run_id": run_id,
                "sharpe_ratio": sharpe_ratio,
                "annual_return": annual_return,
                "runtime": runtime,
                **{k: v for k, v in config.items() if k not in ["algorithm", "quality"]}
            })

            # Store histories if valid
            if fitness_history and all(np.isfinite(fitness_history)):
                fitness_histories.append(fitness_history)
            if diversity_history and all(np.isfinite(diversity_history)):
                diversity_histories.append(diversity_history)

        except Exception as e:
            print(f"❌ Error during run {run_id} of {algo} - {quality}: {e}")

    # Save aggregated histories
    if fitness_histories or diversity_histories:
        # Calculate mean and std for histories
        mean_fitness = np.mean(
            fitness_histories, axis=0) if fitness_histories else []
        std_fitness = np.std(
            fitness_histories, axis=0) if fitness_histories else []
        mean_diversity = np.mean(
            diversity_histories, axis=0) if diversity_histories else []
        std_diversity = np.std(diversity_histories,
                               axis=0) if diversity_histories else []

        # Save to files
        results_dir = os.path.join(
            os.getcwd(), "experiments", "results", algo, quality)
        os.makedirs(results_dir, exist_ok=True)

        if len(mean_fitness) > 0:
            fitness_file = os.path.join(
                results_dir, f"aggregated_{quality}_fitness.csv")
            pd.DataFrame({
                "Generation": range(len(mean_fitness)),
                "Mean Fitness": mean_fitness,
                "Std Fitness": std_fitness
            }).to_csv(fitness_file, index=False)

        if len(mean_diversity) > 0:
            diversity_file = os.path.join(
                results_dir, f"aggregated_{quality}_diversity.csv")
            pd.DataFrame({
                "Generation": range(len(mean_diversity)),
                "Mean Diversity": mean_diversity,
                "Std Diversity": std_diversity
            }).to_csv(diversity_file, index=False)

    return pd.DataFrame(results) if results else pd.DataFrame()


def run_selected_configs_parallel(selected_configs, num_runs, dataset, num_companies, risk_free_rate,
                                  start_date, end_date, correlation_level, seed, max_workers=None):
    """
    Run selected configurations in parallel and combine results.

    Args:
        selected_configs (list): List of configurations to run
        num_runs (int): Number of runs per configuration
        dataset (str): Path to dataset
        num_companies (int): Number of companies to include
        risk_free_rate (float): Annual risk-free rate
        start_date (str): Start date for analysis
        end_date (str): End date for analysis
        correlation_level (str): Correlation level (low, medium, high)
        seed (int): Random seed
        max_workers (int, optional): Maximum number of parallel workers

    Returns:
        DataFrame: Combined results from all configurations
    """
    # Filter out None configs (in case select_configs returned None for some)
    valid_configs = [
        config for config in selected_configs if config is not None]

    if not valid_configs:
        print("No valid configurations to run.")
        return pd.DataFrame()

    # Run configurations in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for config in valid_configs:
            futures.append(executor.submit(
                run_single_config,
                config, num_runs, dataset, num_companies, risk_free_rate,
                start_date, end_date, correlation_level, seed
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Running selected configs"):
            try:
                result_df = future.result()
                if not result_df.empty:
                    all_results.append(result_df)
            except Exception as e:
                print(f"Error in parallel run_configuration: {str(e)}")

    # Combine all results
    if not all_results:
        print("No results were generated from any configuration.")
        return pd.DataFrame()

    final_results_df = pd.concat(all_results, ignore_index=True)

    # Save the combined results
    result_path = get_results_path("final_fine_tuning_results.csv")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    final_results_df.to_csv(result_path, index=False)
    print(f"✅ Final fine-tuning results saved to '{result_path}'")

    return final_results_df
