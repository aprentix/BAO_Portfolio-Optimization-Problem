from experiments.problem.experiment_executor import ExperimentExecutor
from pop import PortfolioOptimization

class PSOExperimentExecutor(ExperimentExecutor):
    """
    A class to execute Particle Swarm Optimization (PSO) experiments.
    """

    # may be we need to pass subpart of dates like np.array of shape ratio data for minimaze operations I/O
    def run_single_experiment(self, config: dict, seed: int) -> dict:
        """
        Run a single PSO experiment.

        Args:
            config (dict): Experiment configuration.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: Results of the experiment.
        """

        # need to finish
        problem: PortfolioOptimization = PortfolioOptimization(...)

        problem.optimize(...)

        ## new to funish

        return {
            'experiment_id': config.get('experiment_id', 'unknown'),
            **config,
            'seed': seed,
            'sharpe_ratio': solution.fitness,
            'convergence_history': pso.best_fitness_history,
        }
