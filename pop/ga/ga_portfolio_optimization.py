from inspyred import ec
from random import Random


class GAPortfolioOptimization:
    def __init__(self, **kwargs):
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
        best_fitness = max(population).fitness
        self.best_fitness_history.append(best_fitness)

    def run(self, seed=None):
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
        return best.candidate, best.fitness
