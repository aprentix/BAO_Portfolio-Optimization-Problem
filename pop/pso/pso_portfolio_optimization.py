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
        self.pop_size = kwargs.get('pop_size', 100)
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
        self.portfolio_repair = kwargs.get('portfolio_repair', REPAIR_METHODS_PSO['normalize'])
        
        # State tracking
        self.best_fitness_history = []
        self.diversity_history = []
        self.current_iteration = 0

        self._validate_parameters()

    def _validate_parameters(self):
        """Ensure parameters are within valid ranges"""
        if self.pop_size < 5:
            raise ValueError("Population size must be at least 5")
        if not 0 <= self.w <= 2:
            raise ValueError("Inertia weight (w) should be between 0 and 2")
        if not 0 <= self.c1 <= 4 or not 0 <= self.c2 <= 4:
            raise ValueError("Cognitive and social coefficients should be between 0 and 4")

    def default_generator(self, random, args):
        """Generate valid initial portfolio weights"""
        weights = np.array([random.random() for _ in range(args.get('num_assets', 10))])
        return self.portfolio_repair(weights, args).tolist()

    def history_observer(self, population, num_generations, num_evaluations, args):
        """Track optimization progress and adapt parameters"""
        best = max(population)
        self.best_fitness_history.append(best.fitness)
        self.current_iteration = num_generations
        
        # Track population diversity
        positions = np.array([p.candidate for p in population])
        self.diversity_history.append(np.mean(np.std(positions, axis=0)))
        
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
        pso.terminator = swarm.terminators.generation_termination
        pso.observer = self.history_observer
        
        # Configure PSO parameters
        pso.inertia = self.w
        pso.cognitive_rate = self.c1
        pso.social_rate = self.c2
        pso.velocity_clamp = self.velocity_clamp

        # Wrap components for constraint handling
        def wrapped_generator(random, args):
            return self.generator(random, args)

        def wrapped_evaluator(candidates, args):
            # Batch repair and evaluation
            repaired = [self.portfolio_repair(c, args) for c in candidates]
            return [self.evaluator(c) for c in repaired]

        # Evolve swarm
        final_pop = pso.evolve(
            generator=wrapped_generator,
            evaluator=wrapped_evaluator,
            pop_size=self.pop_size,
            maximize=True,
            bounder=self.bounder,
            max_generations=self.max_iterations,
            num_variables=self.bounder.upper.size,  # Auto-detect dimension
            # Additional parameters
            velocity_clamp=self.velocity_clamp,
            neighborhood_size=int(self.pop_size * 0.2)
        )

        # Return best solution
        best = max(final_pop)
        return Solution(best.candidate, best.fitness)

    @property
    def convergence_report(self):
        """Generate swarm analysis report"""
        return {
            'iterations': self.current_iteration,
            'final_fitness': self.best_fitness_history[-1],
            'final_diversity': self.diversity_history[-1],
            'max_fitness': max(self.best_fitness_history),
            'avg_iteration_time': np.mean(np.diff(self.best_fitness_history))
        }