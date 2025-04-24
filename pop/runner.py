from dataset import DatasetManager
from benchmarks import PortfolioOptimization
from pandas import DataFrame
from typing import List


def runner(dataset_folder_name, n_companies: int, risk_free_rate_annual: float, start_date: str, end_date: str, **kwarg):
    dataset_manager = DatasetManager(dataset_folder_name)

    correlation_level = kwarg.get('correlation_level')

    sharpe_ratios: DataFrame = None
    meta: List[str] = []
    if correlation_level is not None:
        _, _, sharpe_ratios, meta = dataset_manager.read_annual_resume_same_level_correlation(correlation_level,
                                                                                              risk_free_rate_annual, start_date, end_date, n_companies)
    else:
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
        num_companies=n_companies, sharpe_ratios=sharpe_ratios.to_numpy())

    solution = problem.optimize(algorithm_type="ga")

    print(solution)

    return 0
