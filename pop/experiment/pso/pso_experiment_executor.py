from tqdm import tqdm
import numpy as np
from inspyred import ec
from pop.pso.pso_portfolio_optimization import PSOPortfolioOptimization
from pop.util.solution import Solution
from pop.dataset.dataset_manager import DatasetManager

class PSOExperimentExecutor:
    """
    A class to execute Particle Swarm Optimization (PSO) experiments.
    """

    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.results = []

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
        for experiment in tqdm(experiments, desc="Running All PSO Experiments"):
            repeated_results = self.run_repeated_experiment(experiment, num_runs)
            self.results.extend(repeated_results)

    def run_single_experiment(self, config: dict, seed: int) -> dict:
        """
        Run a single PSO experiment.

        Args:
            config (dict): Experiment configuration.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: Results of the experiment.
        """
        # Load financial data
        returns, sharpe_ratios = self._load_data(config)

        # Configure PSO
        pso = PSOPortfolioOptimization(
            generator=self._weight_generator,
            evaluator=lambda c: np.dot(c, sharpe_ratios),
            bounder=ec.Bounder(0, 1),
            pop_size=config['pop_size'],
            max_iterations=config['max_iterations'],
            w=config['inertia_weight'],
            c1=config['cognitive_rate'],
            c2=config['social_rate'],
            portfolio_repair=Solution.REPAIR_METHODS_PSO[config['repair_method']]
        )

        # Run optimization
        solution = pso.run(seed=seed)

        return {
            **config,
            'seed': seed,
            'sharpe_ratio': solution.fitness,
            'convergence_history': pso.best_fitness_history,
            'best_weights': solution.candidate
        }