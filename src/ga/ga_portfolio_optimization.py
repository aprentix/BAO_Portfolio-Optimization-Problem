import numpy as np
from inspyred import ec, benchmarks
from random import Random

class PortfolioOptimization(benchmarks.Benchmark):
    def __init__(self, num_assets, mean_returns, cov_matrix, risk_free_rate=0.042):
        super().__init__(num_assets)
        self.num_assets = num_assets
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.maximize = True
        self.bounder = ec.RealBounder([0.0] * num_assets, [1.0] * num_assets)

    def generator(self, random, args):
        weights = [random.random() for _ in range(self.num_assets)]
        total = sum(weights)
        return [w / total for w in weights]

    def evaluator(self, candidates, args):
        fitness = []
        for candidate in candidates:
            weights = np.array(candidate)
            # Check constraints (sum=1, no negatives)
            if not np.isclose(sum(weights), 1.0) or (weights < 0).any():
                fitness.append(-np.inf)
                continue
            # Calculate Sharpe Ratio
            port_return = np.dot(weights, self.mean_returns)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            if port_volatility == 0:
                sharpe_ratio = 0.0
            else:
                sharpe_ratio = (port_return - self.risk_free_rate) / port_volatility
            fitness.append(sharpe_ratio)
        return fitness

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

    @classmethod 
    def portfolio_repair(random, candidates, args):
        max_weight = 0.1
        max_iterations = 100 # Try to satisfy soft constraint max_weight
        repaired = []
        
        for candidate in candidates:
            weights = np.array(candidate)
            iteration = 0
            valid = False
            
            while not valid and iteration < max_iterations:
                iteration += 1
                weights = np.maximum(weights, 0.0)
                current_sum = np.sum(weights)
                
                if current_sum > 1.0:
                    excess = current_sum - 1.0
                    sorted_indices = np.argsort(-weights)
                    for idx in sorted_indices:
                        if excess <= 0:
                            break
                        deduction = min(excess, weights[idx])
                        weights[idx] -= deduction
                        excess -= deduction
                
                elif current_sum < 1.0:
                    remaining = 1.0 - current_sum
                    idx = random.randint(0, len(weights) - 1)
                    weights[idx] += remaining
                
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
            
            weights = np.maximum(weights, 0.0)
            weights /= np.sum(weights)
            repaired.append(weights.tolist())
        
        return repaired


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