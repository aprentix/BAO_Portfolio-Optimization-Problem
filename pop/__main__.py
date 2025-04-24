"""
Portfolio Optimization Problem (POP)

Portfolio optimization is a fundamental challenge in investment management, focusing on allocating capital among assets to maximize returns while minimizing risk.
This problem is crucial for both institutional investors managing billions of dollars and individuals growing their savings.
"""

import sys

from cli import parse_args
from runner import runner


def main():
    args = parse_args()

    portfolio_shape_ratio, portfolio_annual_return, companies_weights = runner(args.dataset, args.num_companies, args.risk, args.start_day, args.end_day, correlation_level=args.level)

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
