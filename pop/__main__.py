"""
Portfolio Optimization Problem (POP) CLI

Portfolio optimization is a fundamental challenge in investment management, focusing on allocating capital among assets to maximize returns while minimizing risk.
This problem is crucial for both institutional investors managing billions of dollars and individuals growing their savings.
"""

import sys
import os
from pop.cli import parse_args
from pop.runner import runner
from pop.util.print_results import print_results
from pop.util.file_saver import save_results, save_fitness_history, save_diversity_history

def main():
    args = parse_args()

    args_dict = vars(args)

    kwargs = {
        'algorithm_type': args_dict.pop('type'),
        'dataset_folder_name': args_dict.pop('dataset'),
        'num_companies': args_dict.pop('num_companies'),
        'risk_free_rate_annual': args_dict.pop('risk'),
        'start_date': args_dict.pop('start_day'),
        'end_date': args_dict.pop('end_day'),
        'correlation_level': args_dict.pop('level')
    }

    for key, value in args_dict.items():
        if value is not None:
            kwargs[key] = value

    # Run the optimization
    results, fitness_history, diversity_history = runner(**kwargs)

    # Unpack results
    sharpe_ratio, annual_return, weights = results

    # Print results to console
    print_results(sharpe_ratio, annual_return, weights)

    # --- Save Results ---
    # Simplify correlation level
    correlation_str = {
        "low": "L",
        "medium": "M",
        "high": "H",
        None: "N"
    }.get(kwargs['correlation_level'], "N")

    # Simplify parameter string
    if kwargs['algorithm_type'] == "ga":
        param_str = f"ps-{kwargs['pop_size']}_mg-{kwargs['max_generations']}_mr-{kwargs['mutation_rate']}"
    elif kwargs['algorithm_type'] == "pso":
        param_str = f"ss-{kwargs['swarm_size']}_mi-{kwargs['max_iterations']}_w-{kwargs['w']}"

    # Define the results directory relative to the project root
    results_dir = os.path.join("results", kwargs['algorithm_type'])
    base_filename = f"exp_{correlation_str}_{param_str}"

    # Save results
    save_results(results_dir, base_filename, weights, sharpe_ratio, annual_return)

    # Save fitness history if requested
    if args.save_fitness:
        save_fitness_history(results_dir, base_filename, fitness_history)

    # Save diversity history if requested
    if args.save_diversity:
        save_diversity_history(results_dir, base_filename, diversity_history)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
        sys.exit(1)
