import numpy as np
from inspyred import ec
from pop.ga.ga_portfolio_optimization import GAPortfolioOptimization
from pop.util.repair_methods import REPAIR_METHODS_GA
from experiments.problem.experiment_executor import ExperimentExecutor

class GAExperimentExecutor(ExperimentExecutor):
    """
    A class to execute Genetic Algorithm (GA) experiments.
    """

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
            repair_method=REPAIR_METHODS_GA[config['repair_method']],
            tournament_size=config.get('tournament_size', 3),
            num_elites=config.get('num_elites', 1)
        )

        # Run optimization
        solution = ga.run(seed=seed)

        return {
            'experiment_id': config.get('experiment_id', 'unknown'),
            **config,
            'seed': seed,
            'sharpe_ratio': solution.fitness,
            'convergence_history': ga.best_fitness_history,
        }
