import numpy as np
from tqdm import tqdm
from pop.dataset.dataset_manager import DatasetManager
from pop import PortfolioOptimization
from pop.util.solution import Solution


class ExperimentExecutor():
    """
    A base class for executing optimization experiments.
    """

    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.results = []

    def run_single_experiment(self, algorithm_type: str, seed: int, num_companies: int, sharpe_ratios: np.array, **kwargs) -> dict:
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
            algorithm_type=algorithm_type, **kwargs)

        return {
            'experiment_id': kwargs.get('experiment_id', 'unknown'),
            **kwargs,
            'seed': seed,
            'sharpe_ratio': sol.fitness,
            'convergence_history': problem.last_report,
        }

    def run_repeated_experiment(self, algorithm_type: str, experiment: dict, num_runs: int):
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
                result = self.run_single_experiment(algorithm_type=algorithm_type, experiment, seed=run)
                results.append(result)
            except Exception as e:
                print(
                    f"Failed run {run} for experiment {experiment}: {str(e)}")
                continue
        return results

    def run_all_experiments(self, algorithm_type: str, experiments: list[dict], num_runs: int = 1):
        """
        Run all experiments in the list, with an option to repeat each experiment.

        Args:
            experiments (list[dict]): List of experiment configurations.
            num_runs (int): Number of independent runs per experiment.
        """
        for experiment in tqdm(experiments, desc="Running All Experiments"):
            repeated_results = self.run_repeated_experiment(algorithm_type=algorithm_type,
                                                            experiment=experiment, num_runs=num_runs)
            self.results.extend(repeated_results)
