from pandas import DataFrame
from typing import List

from pop.portfolio_optimization import PortfolioOptimization
from pop.dataset.dataset_manager import DatasetManager
from pop.util.solution import Solution

def runner(algorithm_type: str, dataset_folder_name, num_companies: int, risk_free_rate_annual: float, start_date: str, end_date: str, **kwargs) -> tuple[float, float, dict[str, float]]:
    dataset_manager: DatasetManager = DatasetManager(dataset_folder_name)

    correlation_level = kwargs.pop('correlation_level')

    annual_mean_returns: DataFrame = None
    sharpe_ratios: DataFrame = None
    meta: List[str] = []
    if correlation_level is not None:
        annual_mean_returns, _, sharpe_ratios, meta = dataset_manager.read_annual_resume_same_level_correlation(correlation_level,
                                                                                                                risk_free_rate_annual, start_date, end_date, num_companies)
    else:
        annual_mean_returns, _, sharpe_ratios, meta = dataset_manager.read_annual_resume(
            risk_free_rate_annual, start_date, end_date, num_companies)

    if len(meta) < num_companies:
        print(
            f"WARNING: Only {len(meta)} companies found, but {num_companies} were requested.")
        user_input = input(
            "Do you want to continue with the available companies? (y/n): ").strip().lower()

        if user_input != 'y' and user_input != 'yes':
            raise SystemExit("Operation cancelled by user")

    problem: PortfolioOptimization = PortfolioOptimization(
        num_companies=num_companies, sharpe_ratios=sharpe_ratios.to_numpy())

    solution: Solution = problem.optimize(
        algorithm_type,
        **kwargs
    )

    # Return fitness and diversity history along with the solution
    return solution.decode(dataset_manager.get_full_companies_names(meta), annual_mean_returns), problem.fitness_history, problem.diversity_history
