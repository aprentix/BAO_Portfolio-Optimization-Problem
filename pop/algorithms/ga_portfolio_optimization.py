from inspyred import ec
from inspyred.ec import Individual
from random import Random
from pop.util.solution import Solution
from pop.util.repair_methods import REPAIR_METHODS_GA


class GAPortfolioOptimization:
    """
    Genetic Algorithm for Portfolio Optimization

    Features:
    - Default parameter values
    - Proper constraint handling
    - Adaptive mutation rates
    - Input validation
    """

    def __init__(self, **kwargs):
        """
        Initialize with validated parameters and defaults
        """
        # Set default parameters
        self.pop_size = kwargs.get('pop_size', 100)
        self.max_generations = kwargs.get('max_generations', 200)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.gaussian_stdev = kwargs.get('gaussian_stdev', 0.1)
        self.tournament_size = kwargs.get('tournament_size', 3)
        self.num_elites = kwargs.get('num_elites', 1)
        self.selector = kwargs.get(
            'selector', ec.selectors.tournament_selection)
        self.terminator = kwargs.get(
            'terminator', ec.terminators.generation_termination)

        # Portfolio-specific parameters
        self.generator = kwargs.get('generator', self.default_generator)
        self.evaluator = kwargs.get('evaluator')
        self.bounder = kwargs.get('bounder', ec.Bounder(0, 1))
        self.portfolio_repair = kwargs.get(
            'portfolio_repair', REPAIR_METHODS_GA['normalize'])

        # State tracking
        self.best_fitness_history = []
        self.current_generation = 0

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """Ensure parameters are within valid ranges"""
        if self.pop_size < 2:
            raise ValueError("Population size must be at least 2")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if self.tournament_size < 1:
            raise ValueError("Tournament size must be at least 1")

    def default_generator(self, random, args):
        """Default portfolio weight generator"""
        return [random.random() for _ in range(args.get('num_assets', 10))]

    def history_observer(self, population, num_generations, num_evaluations, args):
        """Track best fitness and adapt parameters"""
        best = max(population)
        self.best_fitness_history.append(best.fitness)
        self.current_generation = num_generations

        # Adapt mutation rate
        self.mutation_rate = self._adapt_mutation_rate(num_generations)

    def _adapt_mutation_rate(self, generation):
        """Dynamic mutation rate schedule"""
        return self.mutation_rate * (0.95 ** (generation / 10))

    def run(self, seed=None) -> Solution:
        """Execute GA with constraint handling"""
        rand = Random(seed)

        # Configure GA components
        ga = ec.GA(rand)
        ga.terminator = self.terminator
        ga.observer = self.history_observer
        ga.selector = self.selector
        ga.replacer = ec.replacers.generational_replacement
        ga.maintainer = self.portfolio_repair  # Proper constraint handling

        # Configure variation operators
        ga.variator = [
            ec.variators.blend_crossover,
            ec.variators.gaussian_mutation
        ]

        # Wrap generator and evaluator for constraint management
        def wrapped_generator(random, args):
            candidate = self.generator(random, args)
            return self.portfolio_repair(None, candidate, args)

        def wrapped_evaluator(candidates, args):
            # Batch evaluation for efficiency
            if hasattr(self.evaluator, '__call__'):
                return [self.evaluator(c) for c in candidates]
            raise ValueError("Evaluator must be a callable function")

        # Evolve population
        final_pop = ga.evolve(
            generator=wrapped_generator,
            evaluator=wrapped_evaluator,
            pop_size=self.pop_size,
            maximize=True,
            bounder=self.bounder,
            max_generations=self.max_generations,
            num_elites=self.num_elites,
            blx_alpha=0.5,
            mutation_rate=self.mutation_rate,
            gaussian_stdev=self.gaussian_stdev,
            tournament_size=self.tournament_size,
            # Additional evolutionary parameters
            crowding_distance=10,
            num_selected=self.pop_size
        )

        # Return best solution
        best: Individual = max(final_pop)
        return Solution(best.candidate, best.fitness)

    @property
    def report(self):
        """Generate report"""
        return {
            'generations': self.current_generation,
            'fitness_history': self.best_fitness_history
        }
