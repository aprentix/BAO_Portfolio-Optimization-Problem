from inspyred import swarm, ec
from random import Random
import numpy as np
from pop.util.solution import Solution
from pop.util.repair_methods import REPAIR_METHODS_PSO


class PSOPortfolioOptimization:
    """
    Particle Swarm Optimization for Portfolio Optimization

    Features:
    - Adaptive inertia weight
    - Velocity clamping
    - Input validation
    - Batch evaluation
    - Diversity monitoring
    - Proper constraint handling
    """

    def __init__(self, **kwargs):
        # Set default parameters with validation
        self.swarm_size = kwargs.get('swarm_size', 100)
        self.max_iterations = kwargs.get('max_iterations', 200)
        self.w = kwargs.get('w', 0.7)
        self.c1 = kwargs.get('c1', 1.5)
        self.c2 = kwargs.get('c2', 1.5)
        self.velocity_clamp = kwargs.get('velocity_clamp', 0.5)
        self.adaptive_inertia = kwargs.get('adaptive_inertia', True)

        # Portfolio-specific parameters
        self.generator = kwargs.get('generator', self.default_generator)
        self.evaluator = kwargs.get('evaluator')
        self.bounder = kwargs.get('bounder', ec.Bounder(0, 1))
        self.portfolio_repair = kwargs.get(
            'portfolio_repair', REPAIR_METHODS_PSO['normalize'])

        # State tracking
        self.best_fitness_history = []
        self.diversity_history = []
        self.current_iteration = 0

        self._validate_parameters()

    def _validate_parameters(self):
        """Ensure parameters are within valid ranges"""
        if self.swarm_size < 5:
            raise ValueError("Population size must be at least 5")
        if not 0 <= self.w <= 2:
            raise ValueError("Inertia weight (w) should be between 0 and 2")
        if not 0 <= self.c1 <= 4 or not 0 <= self.c2 <= 4:
            raise ValueError(
                "Cognitive and social coefficients should be between 0 and 4")

    def default_generator(self, random, args):
        """Generate valid initial portfolio weights"""
        weights = np.array([random.random() for _ in range(args.get('num_assets', 10))])
        # Apply repair multiple times to ensure sum to 1
        for _ in range(3):
            weights = self.portfolio_repair(weights, args)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-9), "Weights do not sum to 1 after generation!"
        return weights.tolist()

    def history_observer(self, population, num_generations, num_evaluations, args):
        """Track optimization progress and adapt parameters"""
        best = max(population)
        # Apply repair to best candidate
        repaired = self.portfolio_repair(best.candidate, args)
        for _ in range(2):
            repaired = self.portfolio_repair(repaired, args)
        assert np.isclose(np.sum(repaired), 1.0, atol=1e-9), "Weights do not sum to 1 in observer!"
        # Ensure the best fitness is valid
        best_fitness = best.fitness if np.isfinite(best.fitness) else 0.0
        self.best_fitness_history.append(best_fitness)
        self.current_iteration = num_generations

        # Track population diversity
        positions = np.array([p.candidate for p in population])
        diversity = np.mean(np.std(positions, axis=0))
        if not np.isfinite(diversity):
            diversity = 0.0  # Fallback to 0 if invalid
        self.diversity_history.append(diversity)

        # Adapt inertia weight
        if self.adaptive_inertia:
            self.w = self._adapt_inertia(num_generations)

    def _adapt_inertia(self, iteration):
        """Linearly decreasing inertia weight"""
        initial_w = 0.9
        final_w = 0.4
        return initial_w - (initial_w - final_w) * (iteration / self.max_iterations)

    def run(self, seed=None) -> Solution:
        """Execute PSO with constraint handling"""
        rand = Random(seed)

        # Configure PSO components
        pso = swarm.PSO(rand)
        pso.topology = swarm.topologies.star_topology
        pso.terminator = ec.terminators.generation_termination
        pso.observer = self.history_observer

        # Configure PSO parameters
        pso.inertia = self.w
        pso.cognitive_rate = self.c1
        pso.social_rate = self.c2
        pso.velocity_clamp = self.velocity_clamp

        # Wrap components for constraint handling
        def wrapped_generator(random, args):
            weights = self.generator(random, args)
            for _ in range(3):
                weights = self.portfolio_repair(weights, args)
            # Sanitize weights after repair to avoid NaN or Inf
            weights = np.nan_to_num(weights, nan=1.0 / len(weights), posinf=1.0 / len(weights), neginf=1.0 / len(weights))
            if not np.isclose(np.sum(weights), 1.0, atol=1e-9):
                weights = np.ones(len(weights)) / len(weights)  # Fallback to equal distribution if invalid
            return weights

        def wrapped_evaluator(candidates, args):
            repaired = []
            for c in candidates:
                w = c
                for _ in range(3):
                    w = self.portfolio_repair(w, args)
                # Handle NaN values after repair
                w = np.nan_to_num(w, nan=1.0 / len(w), posinf=1.0 / len(w), neginf=1.0 / len(w))
                if not np.isclose(np.sum(w), 1.0, atol=1e-9):
                    w = np.ones(len(w)) / len(w)  # Fallback to equal distribution if still invalid
                repaired.append(w)

            # Evaluate and sanitize fitness values
            fitness = self.evaluator(repaired)
            # Ensure fitness is finite, otherwise set to a safe default (e.g., 0.0)
            fitness = [0.0 if not np.isfinite(f) else f for f in fitness]
            return fitness



        # Determine the number of variables (dimensions)
        if hasattr(self.bounder, 'lower_bound') and hasattr(self.bounder.lower_bound, '__len__'):
            num_variables = len(self.bounder.lower_bound)
        else:
            raise ValueError(
                "Bounder must have a 'lower_bound' attribute with a defined length.")

        # Evolve swarm
        final_pop = pso.evolve(
            generator=wrapped_generator,
            evaluator=wrapped_evaluator,
            pop_size=self.swarm_size,
            maximize=True,
            bounder=self.bounder,
            max_generations=self.max_iterations,
            num_variables=num_variables,
            # Additional parameters
            velocity_clamp=self.velocity_clamp,
            neighborhood_size=int(self.swarm_size * 0.2)
        )

        # Return best solution, repaired
        best = max(final_pop)
        weights = best.candidate
        for _ in range(3):
            weights = self.portfolio_repair(weights, {})
        assert np.isclose(np.sum(weights), 1.0, atol=1e-9), "Weights do not sum to 1 in final solution!"
        return Solution(weights, best.fitness)

    @property
    def report(self) -> dict:
        """Generate analysis report"""
        return {
            'iterations': self.current_iteration,
            'fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history
        }
