import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from pop.dataset.dataset_manager import DatasetManager
from pop import PortfolioOptimization
from pop.util.solution import Solution


class ExperimentExecutor():
    """
    A base class for executing optimization experiments.
    """

    def run_single_experiment(self, algorithm_type: str, seed: int, num_companies: int, sharpe_ratios: np.array, experiment: dict) -> dict:
        """
        Run a single experiment. Must be implemented by subclasses.

        Args:
            config (dict): Experiment configuration.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: Results of the experiment.
        """
        problem: PortfolioOptimization = PortfolioOptimization(
            num_companies=num_companies, sharpe_ratios=sharpe_ratios)

        sol: Solution = problem.optimize(
            algorithm_type=algorithm_type, kwargs=experiment)

        return {
            'experiment_id': experiment.get('experiment_id', 'unknown'),
            **experiment,
            'seed': seed,
            'sharpe_ratio': sol.fitness,
            # 'convergence_history': problem.last_report,
        }

    def run_repeated_experiment(self, algorithm_type: str, seed: int, num_companies: int, sharpe_ratios: np.array, experiment: dict, num_runs: int):
        """
        Run a single experiment multiple times and aggregate results.

        Args:
            experiment (dict): Experiment configuration.
            num_runs (int): Number of independent runs.

        Returns:
            list[dict]: List of results for each run.
        """
        results = []
        for run in range(num_runs):
            try:
                result = self.run_single_experiment(
                    algorithm_type=algorithm_type,
                    seed=seed,
                    num_companies=num_companies,
                    sharpe_ratios=sharpe_ratios,
                    experiment=experiment
                )
                results.append(result)
            except Exception as e:
                print(
                    f"Failed run {run} for experiment {experiment}: {str(e)}")
                continue
        return results

    def run_all_experiments(self, dataset_manager: DatasetManager, risk_free_rate_annual: float, start_date: str, end_date: str, algorithm_type: str, experiments: list[dict], seed: int, num_runs: int = 1):
        """
        Run all experiments in the list, with an option to repeat each experiment.

        Args:
            experiments (list[dict]): List of experiment configurations.
            num_runs (int): Number of independent runs per experiment.
        """
        results = []

        experiments_groups: dict[str, list[dict]] = self.__group_by_correlation_level(
            experiments=experiments)

        for group in experiments_groups:
            sharpe_ratios: DataFrame = None
            if group != 'None':
                _, _, sharpe_ratios, _ = dataset_manager.read_annual_resume_same_level_correlation(
                    group, risk_free_rate_annual, start_date, end_date, n_companies=experiments_groups[group][-1].get('num_assets'))
            else:
                _, _, sharpe_ratios, _ = dataset_manager.read_annual_resume(
                    risk_free_rate_annual, start_date, end_date, n_companies=experiments_groups[group][-1].get('num_assets'))

            for experiment in tqdm(experiments_groups[group], desc=f"Experiments with correlation level {group}"):
                repeated_results = self.run_repeated_experiment(
                    algorithm_type=algorithm_type,
                    seed=seed,
                    num_companies=experiment.get('num_assets'),
                    sharpe_ratios=sharpe_ratios.head(
                        experiment.get('num_assets')).to_numpy(),
                    experiment=experiment,
                    num_runs=num_runs
                )
                results.extend(repeated_results)
        return results

    def __group_by_correlation_level(self, experiments: list[dict]) -> dict[str, list[dict]]:
        grouped_experiments = {}
        for exp in experiments:
            corr_level = exp.get('correlation_level')
            if corr_level not in grouped_experiments:
                grouped_experiments[corr_level] = []
            grouped_experiments[corr_level].append(exp)

        for corr_level in grouped_experiments:
            grouped_experiments[corr_level].sort(
                key=lambda x: x.get("num_assets"))

        return grouped_experiments
