import os
import pandas as pd
import numpy as np
from itertools import product
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from pop.runner import runner
from pop.util.file_saver import (
    prepare_file_saving, save_results, save_fitness_history, save_diversity_history
)

def get_project_root():
    """
    Returns the absolute path to the project root (parent directory containing 'pop').
    """
    current_dir = os.path.abspath(os.getcwd())
    while True:
        if "pop" in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not find project root containing 'pop' directory.")
        current_dir = parent_dir

def get_results_path(filename, algorithm=None):
    """
    Returns the path to the results file, optionally within an algorithm subfolder.
    """
    project_root = get_project_root()
    results_dir = os.path.join(project_root, "experiments", "results")
    if algorithm:
        return os.path.join(results_dir, algorithm, filename)
    return os.path.join(results_dir, filename)

def check_existing_results(results_dir, base_filename, num_runs):
    completed_runs = 0
    for run_id in range(1, num_runs + 1):
        run_filename = os.path.join(results_dir, f"{base_filename}_run{run_id}_results.csv")
        if os.path.isfile(run_filename):
            completed_runs += 1
    return completed_runs

def get_param_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, combo)) for combo in product(*values)]

def evaluate_config(algorithm_type, config, repair_method, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed):
    scores, times, returns, generations = [], [], [], []
    results_dir, base_filename = prepare_file_saving(
        algorithm_type=algorithm_type,
        correlation_level=correlation_level,
        params=config,
        root_path=os.path.join(get_project_root(), "experiments", "results")
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
            save_results(results_dir, run_filename, weights, sharpe_ratio, annual_return)
            save_fitness_history(results_dir, run_filename, fitness_history)
            save_diversity_history(results_dir, run_filename, diversity_history)
            print(f"✅ Run {run_id}/{num_runs} completed for {algorithm_type} - {config}")
        except Exception as e:
            print(f"Error during evaluation of {algorithm_type} with config {config}: {e}")
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_return = np.mean(returns)
    mean_time = np.mean(times)
    mean_generations = np.mean(generations)
    aggregated_filename = f"{base_filename}_aggregated"
    save_results(results_dir, aggregated_filename, weights, mean_score, mean_return)
    save_fitness_history(results_dir, aggregated_filename, fitness_history)
    save_diversity_history(results_dir, aggregated_filename, diversity_history)
    return mean_score, std_score, mean_return, mean_time, mean_generations, config

def fine_tune_algorithms(num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed, ga_param_grid, pso_param_grid, REPAIR_METHODS):
    results = []
    for algo_type, param_grid in [("GA", ga_param_grid), ("PSO", pso_param_grid)]:
        print(f"Starting fine-tuning for {algo_type}...")
        for repair in REPAIR_METHODS:
            for config in get_param_combinations(param_grid):
                print(f"Testing {algo_type} with config: {config} + repair method: {repair}")
                mean_score, std_score, mean_return, mean_time, mean_generations, used_config = evaluate_config(
                    algo_type.lower(), config, repair, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed)
                results.append({
                    "algorithm": algo_type,
                    "repair_method": repair,
                    "mean_sharpe": mean_score,
                    "std_sharpe": std_score,
                    "mean_return": mean_return,
                    "mean_time": mean_time,
                    "mean_generations": mean_generations,
                    **used_config
                })
    for algo_type in ["GA", "PSO"]:
        algo_results = [r for r in results if r['algorithm'] == algo_type]
        df = pd.DataFrame(algo_results)
        result_path = get_results_path("fine_tuning_results.csv", algo_type.lower())
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        df.to_csv(result_path, index=False)
        print(f"✅ Fine-tuning results for {algo_type} saved to '{result_path}'")
    return results

def load_fine_tuning_results():
    ga_file_path = get_results_path("fine_tuning_results.csv", "ga")
    pso_file_path = get_results_path("fine_tuning_results.csv", "pso")
    try:
        print(f"🔍 Loading GA results from: {ga_file_path}")
        ga_results = pd.read_csv(ga_file_path)
    except FileNotFoundError:
        print(f"⚠️ GA fine-tuning results not found at {ga_file_path}.")
        ga_results = None

    try:
        print(f"🔍 Loading PSO results from: {pso_file_path}")
        pso_results = pd.read_csv(pso_file_path)
    except FileNotFoundError:
        print(f"⚠️ PSO fine-tuning results not found at {pso_file_path}.")
        pso_results = None

    if ga_results is not None and pso_results is not None:
        combined_results = pd.concat([ga_results, pso_results], ignore_index=True)
        print("✅ Combined GA and PSO results loaded successfully.")
        return combined_results
    elif ga_results is not None:
        print("✅ Only GA results loaded successfully.")
        return ga_results
    elif pso_results is not None:
        print("✅ Only PSO results loaded successfully.")
        return pso_results
    else:
        print("⚠️ No fine-tuning results found for either algorithm.")
        return None

def load_history(algorithm, base_filename, quality):
    project_root = get_project_root()
    results_path = os.path.join(project_root, "experiments", "results", algorithm)

    # Look for aggregated fitness and diversity files
    fitness_file = os.path.join(results_path, f"{base_filename}_aggregated_fitness.csv")
    diversity_file = os.path.join(results_path, f"{base_filename}_aggregated_diversity.csv")

    if os.path.exists(fitness_file):
        fitness_history = pd.read_csv(fitness_file)
        print(f"✅ Loaded aggregated fitness history for {algorithm.upper()} - {quality} from {fitness_file}")
    else:
        print(f"❌ No aggregated fitness history found for {algorithm.upper()} - {quality}")
        fitness_history = None

    if os.path.exists(diversity_file):
        diversity_history = pd.read_csv(diversity_file)
        print(f"✅ Loaded aggregated diversity history for {algorithm.upper()} - {quality} from {diversity_file}")
    else:
        print(f"❌ No aggregated diversity history found for {algorithm.upper()} - {quality}")
        diversity_history = None

    return fitness_history, diversity_history

def generate_base_filename(config, algorithm, correlation_level=None):
    correlation_str = {
        "low": "L",
        "medium": "M",
        "high": "H",
        None: "N"
    }.get(correlation_level, "N")

    if algorithm == "ga":
        pop_size = int(config["pop_size"]) if not pd.isna(config["pop_size"]) else "NA"
        max_gen = int(config["max_generations"]) if not pd.isna(config["max_generations"]) else "NA"
        mutation_rate = config["mutation_rate"]
        return f"exp_{correlation_str}_ps-{pop_size}_mg-{max_gen}_mr-{mutation_rate}"
    elif algorithm == "pso":
        swarm_size = int(config["swarm_size"]) if not pd.isna(config["swarm_size"]) else "NA"
        max_iter = int(config["max_iterations"]) if not pd.isna(config["max_iterations"]) else "NA"
        w = config["w"]
        return f"exp_{correlation_str}_ss-{swarm_size}_mi-{max_iter}_w-{w}"
    else:
        return "unknown_config"

def print_configuration_details(config, algo, config_name):
    print(f"🏆 {config_name.capitalize()} configuration for {algo.upper()}:")
    for key, value in config.items():
        if pd.notna(value):
            print(f"  {key}: {value}")
    print("\n")

def aggregate_history(history_list):
    filtered_history_list = [h for h in history_list if len(h) > 0 and np.isfinite(np.sum(h))]
    if len(filtered_history_list) == 0:
        return [], []
    history_array = np.array(filtered_history_list)
    mean_history = np.nanmean(history_array, axis=0)
    std_history = np.nanstd(history_array, axis=0)
    return mean_history, std_history

def save_aggregated_history(results_dir, filename, mean_history, std_history, metric):
    if len(mean_history) == 0:
        print(f"⚠️ No valid aggregated {metric} history to save.")
        return
    aggregated_file = os.path.join(results_dir, f"{filename}_aggregated_{metric}.csv")
    aggregated_df = pd.DataFrame({
        "Generation": range(len(mean_history)),
        f"Mean {metric.capitalize()}": mean_history,
        f"Std {metric.capitalize()}": std_history
    })
    aggregated_df.to_csv(aggregated_file, index=False)
    print(f"✅ Aggregated {metric} history saved to {aggregated_file}")

def run_configuration(algorithm, config, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed):
    results = []
    fitness_histories = []
    diversity_histories = []
    for run_id in range(1, num_runs + 1):
        try:
            runner_params = config.to_dict()
            int_params = ["pop_size", "num_elites", "max_generations", "swarm_size", "max_iterations"]
            for param in int_params:
                if param in runner_params and not pd.isna(runner_params[param]):
                    runner_params[param] = int(runner_params[param])
            if "repair_method" in runner_params:
                del runner_params["repair_method"]
            start = time.time()
            run_output, fitness_history, diversity_history = runner(
                algorithm_type=algorithm.lower(),
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
            sharpe_ratio, annual_return, weights = run_output
            if not np.isfinite(sharpe_ratio) or not np.isfinite(annual_return):
                continue
            if not fitness_history or not all(np.isfinite(fitness_history)):
                continue
            if not diversity_history or not all(np.isfinite(diversity_history)):
                continue
            runtime = time.time() - start
            results.append({
                "algorithm": algorithm,
                "quality": config["quality"],
                "run_id": run_id,
                "sharpe_ratio": sharpe_ratio,
                "annual_return": annual_return,
                "runtime": runtime,
                **config.to_dict()
            })
            fitness_histories.append(fitness_history)
            diversity_histories.append(diversity_history)
        except Exception as e:
            print(f"❌ Error during run {run_id} of {algorithm} - {config['quality']}: {e}")
    mean_fitness, std_fitness = aggregate_history(fitness_histories)
    mean_diversity, std_diversity = aggregate_history(diversity_histories)
    results_dir = get_results_path(f"{algorithm.lower()}/{config['quality']}")
    os.makedirs(results_dir, exist_ok=True)
    filename = f"aggregated_{config['quality']}"
    save_aggregated_history(results_dir, filename, mean_fitness, std_fitness, "fitness")
    save_aggregated_history(results_dir, filename, mean_diversity, std_diversity, "diversity")
    print(f"Returning DataFrame with {len(results)} results for {config['quality']} {algorithm}")
    return pd.DataFrame(results)

# Function to find the best, median, and worst configurations
def select_configs(df, algorithm):
    df_algo = df[df["algorithm"] == algorithm]

    # Sort the dataframe by mean_sharpe in descending order for clarity
    df_algo = df_algo.sort_values(by="mean_sharpe", ascending=False).reset_index(drop=True)

    # Select best, median, and worst configurations
    best_config = df_algo.iloc[0].copy()
    median_config = df_algo.iloc[len(df_algo) // 2].copy()
    worst_config = df_algo.iloc[-1].copy()

    # Add the quality attribute to each configuration
    best_config["quality"] = "best"
    median_config["quality"] = "median"
    worst_config["quality"] = "worst"

    return best_config, median_config, worst_config

def run_selected_configs(fine_tuning_results, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed):
    all_results = []
    for algo in ["GA", "PSO"]:
        best, median, worst = select_configs(fine_tuning_results, algo)
        for config in [best, median, worst]:
            print(f"🚀 Running {algo} - {config['quality']} configuration...")
            result_df = run_configuration(algo, config, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed)
            all_results.append(result_df)
    final_results_df = pd.concat(all_results, ignore_index=True)
    result_path = get_results_path("final_fine_tuning_results.csv")
    final_results_df.to_csv(result_path, index=False)
    print(f"✅ Final fine-tuning results saved to '{result_path}'")
    return final_results_df

def fine_tune_algorithms_parallel(num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed, ga_param_grid, pso_param_grid, REPAIR_METHODS, max_workers=None):
    jobs = []
    for algo_type, param_grid in [("GA", ga_param_grid), ("PSO", pso_param_grid)]:
        for repair in REPAIR_METHODS:
            for config in get_param_combinations(param_grid):
                jobs.append((algo_type.lower(), config, repair, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed))
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_config, *job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fine-tuning configs"):
            try:
                mean_score, std_score, mean_return, mean_time, mean_generations, used_config = future.result()
                results.append({
                    "algorithm": used_config.get("algorithm", used_config.get("algorithm_type", None) or "GA" if jobs[0][0] == "ga" else "PSO"),
                    "repair_method": used_config.get("repair_method", None),
                    "mean_sharpe": mean_score,
                    "std_sharpe": std_score,
                    "mean_return": mean_return,
                    "mean_time": mean_time,
                    "mean_generations": mean_generations,
                    **used_config
                })
            except Exception as e:
                print(f"Error in parallel evaluation: {e}")
    for algo_type in ["GA", "PSO"]:
        algo_results = [r for r in results if r['algorithm'] == algo_type]
        df = pd.DataFrame(algo_results)
        result_path = get_results_path("fine_tuning_results.csv", algo_type.lower())
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        df.to_csv(result_path, index=False)
        print(f"✅ Fine-tuning results for {algo_type} saved to '{result_path}'")
    return results

def run_single_config(config, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed):
    if "repair_method" not in config or pd.isna(config["repair_method"]) or config["repair_method"] == "":
        config = config.copy()
        config["repair_method"] = "normalize"
    algo = config["algorithm"]
    return run_configuration(
        algo, config, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed
    )

def run_selected_configs_parallel(selected_configs, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed, max_workers=None):
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_config, config, num_runs, dataset, num_companies, risk_free_rate, start_date, end_date, correlation_level, seed)
            for config in selected_configs
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Selected configs"):
            try:
                result_df = future.result()
                all_results.append(result_df)
            except Exception as e:
                print(f"Error in parallel run_configuration: {e}")
    final_results_df = pd.concat(all_results, ignore_index=True)
    result_path = get_results_path("final_fine_tuning_results.csv")
    final_results_df.to_csv(result_path, index=False)
    print(f"✅ Final fine-tuning results saved to '{result_path}'")
    return final_results_df