import numpy as np
from inspyred import ec, benchmarks
from ga.ga_portfolio_optimization import GAPortfolioOptimization
from util.solution import Solution


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
        self.num_companies = num_companies
        self.sharpe_ratios = sharpe_ratios
        self.bounder = ec.Bounder(
            [0.0] * num_companies, [1.0] * num_companies)

    def generator(self, random, args):
        """
        Generate a random initial portfolio allocation that sums to 1.

        Args:
            random: Random number generator.
            args: Additional arguments.

        Returns:
            list: A list of normalized random weights that sum to 1.
        """
        weights = [random.random() for _ in range(self.num_companies)]
        total = sum(weights)
        return [w / total for w in weights]

    def evaluator(self, candidates, args):
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
            # Check constraints (sum=1, no negatives)
            if not np.isclose(sum(weights), 1.0) or (weights < 0).any():
                fitness.append(-np.inf)
                continue

            portfolio_sharpe_ratio = np.sum(self.sharpe_ratios * weights)

            fitness.append(portfolio_sharpe_ratio)
        return fitness

    @classmethod
    def portfolio_repair(random, candidates, args):
        """
        Repair operator that ensures portfolio allocations satisfy constraints.

        This method modifies invalid portfolios to satisfy the following constraints:
        1. All weights are non-negative
        2. Weights sum to 1.0
        3. No individual weight exceeds max_weight (0.1 or 10%)

        Args:
            random: Random number generator.
            candidates: List of candidate portfolio allocations to repair.
            args: Additional arguments.

        Returns:
            list: Repaired candidate portfolio allocations.
        """
        max_weight = 0.1  # Maximum 10% allocation to any single asset
        max_iterations = 100  # Try to satisfy soft constraint max_weight
        repaired = []

        for candidate in candidates:
            weights = np.array(candidate)
            iteration = 0
            valid = False

            while not valid and iteration < max_iterations:
                iteration += 1
                # Ensure non-negative weights
                weights = np.maximum(weights, 0.0)
                current_sum = np.sum(weights)

                # Handle case where sum > 1: reduce largest weights first
                if current_sum > 1.0:
                    excess = current_sum - 1.0
                    sorted_indices = np.argsort(-weights)
                    for idx in sorted_indices:
                        if excess <= 0:
                            break
                        deduction = min(excess, weights[idx])
                        weights[idx] -= deduction
                        excess -= deduction

                # Handle case where sum < 1: add remainder to a random weight
                elif current_sum < 1.0:
                    remaining = 1.0 - current_sum
                    idx = random.randint(0, len(weights) - 1)
                    weights[idx] += remaining

                # Handle case where weights exceed max_weight
                over_indices = np.where(weights > max_weight)[0]
                if len(over_indices) > 0:
                    excess = np.sum(weights[over_indices] - max_weight)
                    weights[over_indices] = max_weight
                    under_indices = np.where(weights < max_weight)[0]
                    if len(under_indices) > 0:
                        remaining = excess
                        while remaining > 1e-6:
                            idx = random.choice(under_indices)
                            available_space = max_weight - weights[idx]
                            add_amount = min(available_space, remaining)
                            weights[idx] += add_amount
                            remaining -= add_amount
                            if weights[idx] >= max_weight - 1e-6:
                                under_indices = under_indices[under_indices != idx]
                                if len(under_indices) == 0:
                                    break
                else:
                    if np.isclose(np.sum(weights), 1.0, atol=1e-6):
                        valid = True

            # Final normalization to ensure sum equals 1.0 exactly
            weights = np.maximum(weights, 0.0)
            weights /= np.sum(weights)
            repaired.append(weights.tolist())

        return repaired

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
        match(algorithm_type):
            case "ga":
                return self.__run_ga(
                    generator=self.generator,
                    evaluator=self.evaluator,
                    bounder=self.bounder,
                    pop_size=kwargs.get('pop_size', 100),
                    max_generations=kwargs.get('max_generations', 100),
                    selector=kwargs.get(
                        'selector', ec.selectors.tournament_selection),
                    tournament_size=kwargs.get('tournament_size', 2),
                    mutation_rate=kwargs.get('mutation_rate', 0.1),
                    gaussian_stdev=kwargs.get('gaussian_stdev', 0.1),
                    num_elites=kwargs.get('num_elites', 1),
                    terminator=kwargs.get(
                        'terminator', ec.terminators.generation_termination),
                    portfolio_repair=self.portfolio_repair
                )
            case "pso":
                return self.__run_pso(kwargs)
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
        return ga.run(seed=kwargs.get('seed'))

    def __run_pso(self, **kwargs) -> Solution:
        """
        Run the particle swarm optimization for portfolio allocation.

        Note: This method is currently a placeholder and not implemented.

        Args:
            **kwargs: Parameters for the PSO algorithm.

        Returns:
            Solution: Best solution found by PSO (currently returns None).
        """
        return None
