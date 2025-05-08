from inspyred import ec
import numpy as np
from pop.pso.pso_portfolio_optimization import PSOPortfolioOptimization
from pop.util.repair_methods import REPAIR_METHODS_PSO
from experiments.problem.experiment_executor import ExperimentExecutor

class PSOExperimentExecutor(ExperimentExecutor):
    """
    A class to execute Particle Swarm Optimization (PSO) experiments.
    """

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

        # Determine the number of assets (variables)
        num_assets = len(returns)

        # Configure PSO
        pso = PSOPortfolioOptimization(
            generator=self._weight_generator,
            evaluator=lambda c: np.dot(c, sharpe_ratios),
            bounder=ec.Bounder([0] * num_assets, [1] * num_assets),
            pop_size=config['pop_size'],
            max_iterations=config['max_iterations'],
            w=config['inertia_weight'],
            c1=config['cognitive_rate'],
            c2=config['social_rate'],
            portfolio_repair=REPAIR_METHODS_PSO[config['repair_method']]
        )

        # Run optimization
        solution = pso.run(seed=seed)

        return {
            'experiment_id': config.get('experiment_id', 'unknown'),
            **config,
            'seed': seed,
            'sharpe_ratio': solution.fitness,
            'convergence_history': pso.best_fitness_history,
        }
