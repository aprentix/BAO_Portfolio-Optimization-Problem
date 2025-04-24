from inspyred import ec
from random import Random

from util.solution import Solution


class GAPortfolioOptimization:
    """
    A genetic algorithm implementation for portfolio optimization problems.

    This class uses the inspyred evolutionary computation framework to find optimal 
    asset allocation in a portfolio to maximize a given fitness function (typically 
    risk-adjusted returns).

    Attributes:
        generator: Function that generates initial candidate solutions.
        evaluator: Function that evaluates the fitness of candidate solutions.
        bounder: Function that constrains candidate solutions to valid ranges.
        pop_size: Size of the population in each generation.
        max_generations: Maximum number of generations to evolve.
        selector: Selection method for choosing parents (e.g., tournament selection).
        tournament_size: Number of individuals compared in tournament selection.
        mutation_rate: Probability of mutation for each component of a candidate.
        gaussian_stdev: Standard deviation for Gaussian mutation.
        num_elites: Number of best individuals to preserve between generations.
        terminator: Function that determines when to stop evolution.
        portfolio_repair: Function to ensure portfolio constraints are satisfied.
        best_fitness_history: List tracking the best fitness in each generation.
    """

    def __init__(self, **kwargs):
        """
        Initialize the GA portfolio optimization with customizable parameters.

        Args:
            **kwargs: Keyword arguments:
                - generator: Function to generate initial candidate solutions
                - evaluator: Function to evaluate candidate solutions
                - bounder: Function to constrain solutions to valid ranges
                - pop_size: Population size in each generation
                - max_generations: Maximum number of generations
                - selector: Selection method (e.g., tournament selection)
                - tournament_size: Number of individuals in tournament selection
                - mutation_rate: Probability of mutation
                - gaussian_stdev: Standard deviation for Gaussian mutation
                - num_elites: Number of best individuals to preserve
                - terminator: Function determining when to stop evolution
                - portfolio_repair: Function to ensure portfolio constraints are met
        """
        self.generator = kwargs.get('generator')
        self.evaluator = kwargs.get('evaluator')
        self.bounder = kwargs.get('bounder')
        self.pop_size = kwargs.get('pop_size')
        self.max_generations = kwargs.get('max_generations')
        self.selector = kwargs.get('selector')
        self.tournament_size = kwargs.get('tournament_size')
        self.mutation_rate = kwargs.get('mutation_rate')
        self.gaussian_stdev = kwargs.get('gaussian_stdev')
        self.num_elites = kwargs.get('num_elites')
        self.terminator = kwargs.get('terminator')
        self.portfolio_repair = kwargs.get('portfolio_repair')
        self.best_fitness_history = []

    def history_observer(self, population, num_generations, num_evaluations, args):
        """
        Observer function that tracks the best fitness in each generation.

        This method is called after each generation to record the fitness of the best
        individual in the current population.

        Args:
            population: Current population of candidate solutions
            num_generations: Current generation number
            num_evaluations: Total number of fitness evaluations performed
            args: Additional arguments passed to the evolutionary algorithm
        """
        best_fitness = max(population).fitness
        self.best_fitness_history.append(best_fitness)

    def run(self, seed=None) -> Solution:
        """
        Run the genetic algorithm to find the optimal portfolio allocation.

        This method configures and executes the GA using the inspyred framework,
        applying blend crossover, Gaussian mutation, and the portfolio repair
        operator to evolve solutions toward an optimal portfolio allocation.

        Args:
            seed: Random seed for reproducibility (default: None)

        Returns:
            Solution: A Solution object containing the best candidate (portfolio weights)
                     and its fitness value (typically a risk-adjusted return metric)
        """
        ga = ec.GA(Random(seed))
        ga.terminator = self.terminator
        ga.observer = self.history_observer
        ga.variator = [
            ec.variators.blend_crossover,
            ec.variators.Gaussian_mutation,
            self.portfolio_repair
        ]
        ga.selector = self.selector

        final_pop = ga.evolve(
            generator=self.generator,
            evaluator=self.evaluator,
            pop_size=self.pop_size,
            maximize=True,
            bounder=self.bounder,
            max_generations=self.max_generations,
            num_elites=self.num_elites,
            blx_alpha=0.5,
            mutation_rate=self.mutation_rate,
            gaussian_stdev=self.gaussian_stdev,
            tournament_size=self.tournament_size
        )
        best = max(final_pop)
        return Solution(best.candidate, best.fitness)
