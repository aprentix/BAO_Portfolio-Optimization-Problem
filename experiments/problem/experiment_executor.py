from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from pop.dataset.dataset_manager import DatasetManager

class ExperimentExecutor(ABC):
    """
    A base class for executing optimization experiments.
    """

    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.results = []

    def _load_data(self, config: dict):
        """
        Load financial data required for the experiment.

        Args:
            config (dict): Experiment configuration.

        Returns:
            tuple: Returns (returns, sharpe_ratios) as numpy arrays.
        """
        # Use the dataset manager to load the dataset
        dataset = self.dataset_manager.load_dataset(config['dataset_name'])

        # Extract returns and Sharpe ratios
        returns = dataset['Mean Excess Return'].values
        sharpe_ratios = dataset['Sharpe Ratio'].values

        return returns, sharpe_ratios

    def _weight_generator(self, random, args):
        """
        Generate random weights for the portfolio.

        Args:
            random: Random number generator.
            args: Additional arguments.

        Returns:
            list: A list of weights summing to 1.
        """
        num_assets = args.get('num_assets', 10)  # Default to 10 assets if not specified
        weights = [random.uniform(0, 1) for _ in range(num_assets)]
        total = sum(weights)
        return [w / total for w in weights]

    @abstractmethod
    def run_single_experiment(self, config: dict, seed: int) -> dict:
        """
        Run a single experiment. Must be implemented by subclasses.

        Args:
            config (dict): Experiment configuration.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: Results of the experiment.
        """
        pass

    def run_repeated_experiment(self, experiment: dict, num_runs: int):
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
                result = self.run_single_experiment(experiment, seed=run)
                results.append(result)
            except Exception as e:
                print(f"Failed run {run} for experiment {experiment}: {str(e)}")
                continue
        return results

    def run_all_experiments(self, experiments: list[dict], num_runs: int = 1):
        """
        Run all experiments in the list, with an option to repeat each experiment.

        Args:
            experiments (list[dict]): List of experiment configurations.
            num_runs (int): Number of independent runs per experiment.
        """
        for experiment in tqdm(experiments, desc="Running All Experiments"):
            repeated_results = self.run_repeated_experiment(experiment, num_runs)
            self.results.extend(repeated_results)
