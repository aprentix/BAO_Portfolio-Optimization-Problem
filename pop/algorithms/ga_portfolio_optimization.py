from inspyred import ec
from inspyred.ec import Individual
from random import Random
from pop.util.solution import Solution
from pop.util.repair_methods import REPAIR_METHODS_GA
import numpy as np


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
        self.diversity_history = []
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
        weights = [random.random() for _ in range(args.get('num_assets', 10))]
        return [w / sum(weights) for w in weights]

    def history_observer(self, population, num_generations, num_evaluations, args):
        """Track best fitness and diversity"""
        best = max(population)
        self.best_fitness_history.append(best.fitness)
        self.current_generation = num_generations

        # Calculate diversity as the mean standard deviation of weights
        positions = np.array([ind.candidate for ind in population])
        diversity = np.mean(np.std(positions, axis=0))
        self.diversity_history.append(diversity)

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

        # Custom maintainer to handle portfolio constraints
        def portfolio_maintainer(random, population, args):
            repaired_pop = []
            for ind in population:
                # Repair candidate
                repaired = self.portfolio_repair(random, [ind.candidate], args)[0]
                assert np.isclose(sum(repaired), 1.0, atol=1e-9), "Weights do not sum to 1 after repair!"
                # Create new individual with repaired candidate
                new_ind = Individual(repaired)
                new_ind.fitness = ind.fitness  # Preserve previous fitness
                repaired_pop.append(new_ind)
            return repaired_pop

        ga.maintainer = portfolio_maintainer   

        # Wrap generator and evaluator for constraint management
        def wrapped_generator(random, args):
            candidate = self.generator(random, args)
            repaired = self.portfolio_repair(random, [candidate], args)[0]
            assert np.isclose(sum(repaired), 1.0, atol=1e-9), "Weights do not sum to 1 after generation!"
            return repaired

        def wrapped_evaluator(candidates, args):
            repaired_candidates = [self.portfolio_repair(None, [c], args)[0] for c in candidates]
            for candidate in repaired_candidates:
                assert np.isclose(sum(candidate), 1.0, atol=1e-9), "Weights do not sum to 1 before evaluation!"
            if hasattr(self.evaluator, '__call__'):
                return self.evaluator(repaired_candidates)
            raise ValueError("Evaluator must be a callable function")

        # Wrap variation operators to repair candidates
        def repaired_variator(random, candidates, args):
            varied_candidates = ec.variators.blend_crossover(random, candidates, args)
            varied_candidates = ec.variators.gaussian_mutation(random, varied_candidates, args)
            repaired_candidates = [self.portfolio_repair(random, [c], args)[0] for c in varied_candidates]
            for candidate in repaired_candidates:
                assert np.isclose(sum(candidate), 1.0, atol=1e-9), "Weights do not sum to 1 after variation!"
            return repaired_candidates

        ga.variator = repaired_variator

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
    def report(self) -> dict:
        """Generate report"""
        return {
            'generations': self.current_generation,
            'fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history
        }
