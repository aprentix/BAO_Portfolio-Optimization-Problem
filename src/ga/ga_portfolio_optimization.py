import numpy as np
from inspyred import ec, benchmarks
from random import Random


class GAPortfolioOptimization:
    def __init__(self, num_assets, mean_returns, cov_matrix, **kwargs):
        self.problem = PortfolioOptimization(num_assets, mean_returns, cov_matrix)
        self.pop_size = kwargs.get('pop_size', 100)
        self.max_generations = kwargs.get('max_generations', 100)
        self.selector = kwargs.get('selector', ec.selectors.tournament_selection)
        self.tournament_size = kwargs.get('tournament_size', 2)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.gaussian_stdev = kwargs.get('gaussian_stdev', 0.1)
        self.num_elites = kwargs.get('num_elites', 1)
        self.terminator = kwargs.get('terminator', ec.terminators.generation_termination)
        self.best_fitness_history = []

    def history_observer(self, population, num_generations, num_evaluations, args):
        best_fitness = max(population).fitness
        self.best_fitness_history.append(best_fitness)


    def run(self, seed=None):
        rand = Random(seed)
        ga = ec.GA(rand)
        ga.terminator = self.terminator
        ga.observer = self.history_observer
        ga.variator = [
            ec.variators.blend_crossover,
            ec.variators.Gaussian_mutation,
            GAPortfolioOptimization.portfolio_repair
        ]
        ga.selector = self.selector

        final_pop = ga.evolve(
            generator=self.problem.generator,
            evaluator=self.problem.evaluator,
            pop_size=self.pop_size,
            maximize=True,
            bounder=self.problem.bounder,
            max_generations=self.max_generations,
            num_elites=self.num_elites,
            blx_alpha=0.5,
            mutation_rate=self.mutation_rate,
            gaussian_stdev=self.gaussian_stdev,
            tournament_size=self.tournament_size
        )
        best = max(final_pop)
        return best.candidate, best.fitness