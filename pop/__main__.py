"""
Portfolio Optimization Problem (POP) CLI

Portfolio optimization is a fundamental challenge in investment management, focusing on allocating capital among assets to maximize returns while minimizing risk.
This problem is crucial for both institutional investors managing billions of dollars and individuals growing their savings.
"""

import sys
import os
import pandas as pd
from pop.cli import parse_args
from pop.runner import runner
from pop.util.print_results import print_results

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

    # Save results to a CSV file
    results_dir = os.path.join("results", args.type)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "results.csv")
    pd.DataFrame([
        {"Company": k, "Weight": v, "Percentage": v * 100}
        for k, v in weights.items()
    ]).to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Save fitness history if requested
    if args.save_fitness:
        fitness_file = os.path.join(results_dir, "fitness_history.csv")
        pd.DataFrame({"Generation": range(len(fitness_history)), "Fitness": fitness_history}).to_csv(fitness_file, index=False)
        print(f"Fitness history saved to {fitness_file}")

    # Save diversity history if requested
    if args.save_diversity:
        diversity_file = os.path.join(results_dir, "diversity_history.csv")
        pd.DataFrame({"Generation": range(len(diversity_history)), "Diversity": diversity_history}).to_csv(diversity_file, index=False)
        print(f"Diversity history saved to {diversity_file}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
        sys.exit(1)
