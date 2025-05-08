import numpy as np
from inspyred import ec, benchmarks
from pop.algorithms import GAPortfolioOptimization
from pop.algorithms.pso_portfolio_optimization import PSOPortfolioOptimization
from pop.util.solution import Solution
from pop.util.repair_methods import REPAIR_METHODS_GA, REPAIR_METHODS_PSO

class PortfolioOptimization(benchmarks.Benchmark):
    """
    A portfolio optimization benchmark class that implements various optimization algorithms
    for finding optimal asset allocation based on Sharpe ratios.

    This class extends the inspyred Benchmark class and provides functionality to optimize
    asset allocation within a portfolio to maximize the overall Sharpe ratio, subject to
    constraints such as weights summing to 1, non-negative weights, and maximum weight limits.

    Attributes:
        num_companies (int): Number of assets/companies in the portfolio.
        sharpe_ratios (np.array): Array of Sharpe ratios for each asset.
        bounder (ec.RealBounder): Constrains the search space for each asset weight (0 to 1).
    """

    def __init__(self, num_companies: int, sharpe_ratios: np.array):
        """
        Initialize the PortfolioOptimization benchmark.

        Args:
            num_companies (int): Number of assets/companies in the portfolio.
            sharpe_ratios (np.array): Array of Sharpe ratios for each asset.
        """
        super().__init__(num_companies)
        self._num_companies = num_companies
        self._sharpe_ratios = sharpe_ratios
        self._bounder = ec.Bounder(
            [0.0] * num_companies, [1.0] * num_companies)
        
        self._report: dict | None = None

    def generator(self, random, args):
        """
        Generate a random initial portfolio allocation that sums to 1.

        Args:
            random: Random number generator.
            args: Additional arguments.

        Returns:
            list: A list of normalized random weights that sum to 1.
        """
        weights = [random.random() for _ in range(self._num_companies)]
        total = sum(weights)
        return [w / total for w in weights]

    def evaluator(self, candidates, *args):
        """
        Evaluate the fitness of candidate portfolio allocations based on Sharpe ratio.

        This method checks if portfolio constraints are satisfied and calculates
        the weighted sum of Sharpe ratios for valid portfolios.

        Args:
            candidates: List of candidate portfolio allocations.
            args: Additional arguments.

        Returns:
            list: Fitness values (portfolio Sharpe ratios) for each candidate.
        """
        fitness = []
        for candidate in candidates:
            weights = np.array(candidate)
            max_allocation = np.full_like(weights, 0.1)

            # Check constraints
            if not np.isclose(np.sum(weights), 1.0) or (weights < 0).any():
                fitness.append(-np.inf)
                continue

            # Limits by active
            if (weights > max_allocation).any():
                fitness.append(-np.inf)
                continue

            # Sharpe ratio
            portfolio_sharpe_ratio = np.sum(self._sharpe_ratios * weights)
            fitness.append(portfolio_sharpe_ratio)

        return fitness
    
    def optimize(self, algorithm_type: str, **kwargs) -> Solution:
        """
        Execute the specified optimization algorithm to find optimal portfolio allocation.

        This method supports multiple optimization algorithms, currently implemented:
        - "ga": Genetic Algorithm
        - "pso": Particle Swarm Optimization (placeholder, not yet implemented)

        Args:
            algorithm_type (str): Type of optimization algorithm to use.
            **kwargs: Additional arguments for the specific algorithm:
                - pop_size: Population size
                - max_generations: Maximum number of generations
                - selector: Selection method
                - tournament_size: Size of tournament for selection
                - mutation_rate: Probability of mutation
                - gaussian_stdev: Standard deviation for Gaussian mutation
                - num_elites: Number of best solutions to preserve
                - terminator: Termination criterion
                - seed: Random seed for reproducibility

        Returns:
            Solution: Best solution found containing portfolio weights and fitness.

        Raises:
            ValueError: If an unsupported algorithm type is specified.
        """
        repair_method_name = kwargs.pop("repair_method", "normalize")
        match(algorithm_type):
            case "ga":
                return self.__run_ga(
                    generator=self.generator,
                    evaluator=self.evaluator,
                    bounder=self._bounder,
                    pop_size=kwargs.pop('pop_size', 100),
                    max_generations=kwargs.pop('max_generations', 100),
                    selector=kwargs.pop(
                        'selector', ec.selectors.tournament_selection),
                    tournament_size=kwargs.pop('tournament_size', 2),
                    mutation_rate=kwargs.pop('mutation_rate', 0.1),
                    gaussian_stdev=kwargs.pop('gaussian_stdev', 0.1),
                    num_elites=kwargs.pop('num_elites', 1),
                    terminator=kwargs.pop(
                        'terminator', ec.terminators.generation_termination),
                    portfolio_repair=REPAIR_METHODS_GA.get(repair_method_name),
                    **kwargs
                )
            case "pso":
                return self.__run_pso(
                    generator=self.generator,
                    evaluator=self.evaluator,
                    bounder=self._bounder,
                    swarm_size=kwargs.pop('swarm_size', 100),
                    max_iterations=kwargs.pop('max_iterations', 100),
                    w=kwargs.pop('w', 0.7),
                    c1=kwargs.pop('c1', 1.5),
                    c2=kwargs.pop('c2', 1.5),
                    portfolio_repair=REPAIR_METHODS_PSO.get(repair_method_name),
                    **kwargs
                )
            case _:
                raise ValueError(f"Algorithm {algorithm_type} doesn\'t exist")

    def __run_ga(self, **kwargs) -> Solution:
        """
        Run the genetic algorithm optimization for portfolio allocation.

        Args:
            **kwargs: Parameters for the genetic algorithm.

        Returns:
            Solution: Best solution found by the genetic algorithm.
        """
        ga = GAPortfolioOptimization(**kwargs)
        sol: Solution = ga.run(seed=kwargs.get('seed'))
        self._report = ga.report
        return sol

    def __run_pso(self, **kwargs) -> Solution:
        """
        Run the particle swarm optimization for portfolio allocation.

        Args:
            **kwargs: Parameters for the PSO algorithm.

        Returns:
            Solution: Best solution found by the PSO algorithm.
        """
        pso = PSOPortfolioOptimization(**kwargs)
        sol: Solution = pso.run(seed=kwargs.get('seed'))
        self._report = pso.report
        return sol
    
    @property
    def report(self) -> dict | None:
        return self._report
