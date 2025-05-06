from tqdm import tqdm
import numpy as np
from inspyred import ec
from pop.ga.ga_portfolio_optimization import GAPortfolioOptimization
from pop.util.solution import Solution
from pop.dataset.dataset_manager import DatasetManager
from pop.util.repair_methods import REPAIR_METHODS_GA

class GAExperimentExecutor:
    """
    A class to execute Genetic Algorithm (GA) experiments.
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
                result = self._run_single_experiment(experiment, run)
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
        for experiment in tqdm(experiments, desc="Running All GA Experiments"):
            repeated_results = self.run_repeated_experiment(experiment, num_runs)
            self.results.extend(repeated_results)

    def run_single_experiment(self, config: dict, seed: int) -> dict:
        """
        Run a single GA experiment.

        Args:
            config (dict): Experiment configuration.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: Results of the experiment.
        """
        # Load financial data
        returns, sharpe_ratios = self._load_data(config)

        # Configure GA
        ga = GAPortfolioOptimization(
            generator=self._weight_generator,
            evaluator=lambda c: np.dot(c, sharpe_ratios),
            bounder=ec.Bounder(0, 1),
            pop_size=config['pop_size'],
            max_generations=config['max_generations'],
            mutation_rate=config['mutation_rate'],
            repair_method=REPAIR_METHODS_GA[config['repair_method']],  # Use REPAIR_METHODS_GA directly
            tournament_size=config.get('tournament_size', 3),
            num_elites=config.get('num_elites', 1)
        )

        # Run optimization
        solution = ga.run(seed=seed)

        return {
            **config,
            'seed': seed,
            'sharpe_ratio': solution.fitness,
            'convergence_history': ga.best_fitness_history,
        }
    
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