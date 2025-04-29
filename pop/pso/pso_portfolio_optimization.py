import numpy as np
from random import Random
from util.solution import Solution


def repair_portfolio_pso(weights, max_weight=0.10):
    """
    Repairs a portfolio weights vector by applying:
    - No short-selling (weights >= 0)
    - Max weight per asset
    - Normalization so that sum(weights) == 1
    """
    weights = np.clip(weights, 0, max_weight)
    total = np.sum(weights)
    if total == 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / total
    return weights


class PSOPortfolioOptimization:
    """
    A particle swarm optimization implementation for portfolio optimization problems.
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
        self.portfolio_repair = kwargs.get('portfolio_repair', repair_portfolio_pso)
        self.best_fitness_history = []

    def run(self, seed=None) -> Solution:
        rand = Random(seed)

        # Initialize particles as numpy arrays
        particles = [np.array(self.generator(rand)) for _ in range(self.pop_size)]
        velocities = [np.zeros_like(p) for p in particles]
        personal_best_positions = [np.copy(p) for p in particles]
        personal_best_fitnesses = [self.evaluator(p) for p in particles]

        # Find initial global best
        best_index = np.argmax(personal_best_fitnesses)
        global_best_position = np.copy(personal_best_positions[best_index])
        global_best_fitness = personal_best_fitnesses[best_index]

        for iteration in range(self.max_iterations):
            for i in range(self.pop_size):
                r1 = np.random.uniform(size=len(particles[i]))
                r2 = np.random.uniform(size=len(particles[i]))

                cognitive = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social = self.c2 * r2 * (global_best_position - particles[i])
                velocities[i] = self.w * velocities[i] + cognitive + social

                particles[i] = particles[i] + velocities[i]

                if self.bounder:
                    particles[i] = self.bounder(particles[i], None)

                if self.portfolio_repair:
                    particles[i] = self.portfolio_repair(particles[i])

                fitness = self.evaluator(particles[i])

                if fitness > personal_best_fitnesses[i]:
                    personal_best_positions[i] = np.copy(particles[i])
                    personal_best_fitnesses[i] = fitness

                    if fitness > global_best_fitness:
                        global_best_position = np.copy(particles[i])
                        global_best_fitness = fitness

            self.best_fitness_history.append(global_best_fitness)

        return Solution(global_best_position.tolist(), global_best_fitness)
