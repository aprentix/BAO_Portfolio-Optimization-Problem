from inspyred import swarm
from random import Random
import numpy as np
from util.solution import Solution
from util.repair_methods import repair_normalize, repair_clipped_normalize, repair_random_restart


class PSOPortfolioOptimization:
    """
    A particle swarm optimization implementation for portfolio optimization problems
    using the inspyred.swarm framework.
    """

    def __init__(self, **kwargs):
        self.generator = kwargs.get('generator')
        self.evaluator = kwargs.get('evaluator')
        self.bounder = kwargs.get('bounder')
        self.pop_size = kwargs.get('pop_size')
        self.max_iterations = kwargs.get('max_iterations')
        self.w = kwargs.get('w', 0.7)
        self.c1 = kwargs.get('c1', 1.5)
        self.c2 = kwargs.get('c2', 1.5)
        self.portfolio_repair = kwargs.get('portfolio_repair', repair_normalize)
        self.best_fitness_history = []

    def history_observer(self, population, num_generations, num_evaluations, args):
        """
        Observer function to record the best fitness value in each generation.
        """
        best_fitness = max(population).fitness
        self.best_fitness_history.append(best_fitness)

    def run(self, seed=None) -> Solution:
        rand = Random(seed)

        pso = swarm.PSO(rand)
        pso.topology = swarm.topologies.star_topology
        pso.inertia = self.w
        pso.cognitive_rate = self.c1
        pso.social_rate = self.c2
        pso.observer = self.history_observer

        def wrapped_generator(random, args):
            return self.generator(random, args)

        def wrapped_evaluator(candidates, args):
            results = []
            for c in candidates:
                repaired = self.portfolio_repair(c, args)
                results.append(self.evaluator(repaired))
            return results

        final_pop = pso.evolve(
            generator=wrapped_generator,
            evaluator=wrapped_evaluator,
            pop_size=self.pop_size,
            maximize=True,
            bounder=self.bounder,
            max_generations=self.max_iterations
        )

        best = max(final_pop)
        return Solution(best.candidate, best.fitness)
