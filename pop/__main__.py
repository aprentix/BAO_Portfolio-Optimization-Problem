"""
Portfolio Optimization Problem (POP) CLI

Portfolio optimization is a fundamental challenge in investment management, focusing on allocating capital among assets to maximize returns while minimizing risk.
This problem is crucial for both institutional investors managing billions of dollars and individuals growing their savings.
"""

import sys

from pop.cli import parse_args
from pop.runner import runner


def print_results(sharpe_ratio, annual_return, weights_dict):
    """
    Displays portfolio optimization results in an attractive format in the console.

    Args:
        sharpe_ratio: The Sharpe ratio of the optimized portfolio
        annual_return: The annual return of the optimized portfolio
        weights_dict: Dictionary with weights assigned to each company
    """
    # Print header
    print("\n" + "=" * 80)
    print(" " * 25 + "PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 80)

    # Main metrics
    print(f"\nSharpe Ratio: {sharpe_ratio}")
    print(f"Annual Return: {annual_return} ({annual_return*100}%)")

    # Portfolio distribution
    print("\nPortfolio Distribution:")

    # Create a line format for consistency
    line_format = "{:<45} {:>10} {:>15}"

    # Print the header for the table
    print("-" * 80)
    print(line_format.format("Company", "Weight", "Percentage"))
    print("-" * 80)

    # Sort the weights by value (descending) and print each company
    for company, weight in sorted(weights_dict.items(), key=lambda x: x[1], reverse=True):
        print(line_format.format(
            company[:44] + "..." if len(company) > 44 else company,
            f"{weight:.4f}",
            f"{weight*100:.2f}%"
        ))

    print("-" * 80)

    # Summary information
    total_companies = len(weights_dict)
    # Significant weight positions
    active_positions = sum(1 for w in weights_dict.values() if w > 0.001)

    print(
        f"\nSummary: {active_positions} active positions out of {total_companies} total companies")
    print("=" * 80 + "\n")


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

    portfolio_sharpe_ratio, portfolio_annual_return, companies_weights = runner(
        **kwargs)

    print_results(portfolio_sharpe_ratio,
                  portfolio_annual_return, companies_weights)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
        sys.exit(1)
