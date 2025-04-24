from dataset import DatasetManager
from benchmarks import PortfolioOptimization
import sys


def main():
    n_companies = 5
    risk_free_rate_annual = 0.042
    start_date = "2015-01-01"
    end_date = "2020-01-01"

    dataset_manager = DatasetManager('dataset')

    _, _, sharpe_ratios, meta = dataset_manager.read_annual_resume(
        risk_free_rate_annual, start_date, end_date, n_companies)

    if len(meta) < n_companies:
        print(
            f"WARNING: Only {len(meta)} companies found, but {n_companies} were requested.")
        user_input = input(
            "Do you want to continue with the available companies? (y/n): ").strip().lower()

        if user_input != 'y' and user_input != 'yes':
            print("Operation cancelled by user.")
            return 1

    problem = PortfolioOptimization(
        num_companies=n_companies, sharpe_ratios=sharpe_ratios)

    solution = problem.optimize(algorithm_type="ga")

    print(solution)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
        sys.exit(1)
