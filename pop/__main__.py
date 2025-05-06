"""
Portfolio Optimization Problem (POP) CLI

Portfolio optimization is a fundamental challenge in investment management, focusing on allocating capital among assets to maximize returns while minimizing risk.
This problem is crucial for both institutional investors managing billions of dollars and individuals growing their savings.
"""

import sys

from pop.cli import parse_args
from pop.runner import runner


def main():
    args = parse_args()

    portfolio_shape_ratio, portfolio_annual_return, companies_weights = runner(
        algorithm_type=args.type,
        dataset_folder_name=args.dataset,
        num_companies=args.num_companies,
        risk_free_rate_annual=args.risk,
        start_date=args.start_day,
        end_date=args.end_day,
        correlation_level=args.level)

    print(portfolio_shape_ratio)
    print(portfolio_annual_return)
    print(companies_weights)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
        sys.exit(1)
