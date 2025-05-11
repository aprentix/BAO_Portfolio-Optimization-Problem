"""
Portfolio Optimization Problem (POP) CLI

Portfolio optimization is a fundamental challenge in investment management, focusing on allocating capital among assets to maximize returns while minimizing risk.
This problem is crucial for both institutional investors managing billions of dollars and individuals growing their savings.
"""

import sys

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
