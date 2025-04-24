import numpy as np
from inspyred import ec, benchmarks
from ga.ga_portfolio_optimization import GAPortfolioOptimization
from pop.util.solution import Solution


class PortfolioOptimization(benchmarks.Benchmark):
    def __init__(self, num_companies: int, sharpe_ratios: np.array):
        super().__init__(num_companies)
        self.num_companies = num_companies
        self.sharpe_ratios = sharpe_ratios
        self.bounder = ec.RealBounder(
            [0.0] * num_companies, [1.0] * num_companies)

    def generator(self, random, args):
        weights = [random.random() for _ in range(self.num_companies)]
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

            portfolio_sharpe_ratio = np.sum(self.sharpe_ratios * weights)

            fitness.append(portfolio_sharpe_ratio)
        return fitness

    @classmethod
    def portfolio_repair(random, candidates, args):
        max_weight = 0.1
        max_iterations = 100  # Try to satisfy soft constraint max_weight
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

    def optimize(self, algorithm_type: str, **kwargs) -> Solution:
        match(algorithm_type):
            case "ga":
                return self.__run_ga(
                    generator=self.generator,
                    evaluator=self.evaluator,
                    bounder=self.bounder,
                    pop_size=kwargs.get('pop_size', 100),
                    max_generations=kwargs.get('max_generations', 100),
                    selector=kwargs.get(
                        'selector', ec.selectors.tournament_selection),
                    tournament_size=kwargs.get('tournament_size', 2),
                    mutation_rate=kwargs.get('mutation_rate', 0.1),
                    gaussian_stdev=kwargs.get('gaussian_stdev', 0.1),
                    num_elites=kwargs.get('num_elites', 1),
                    terminator=kwargs.get(
                        'terminator', ec.terminators.generation_termination),
                    portfolio_repair=self.portfolio_repair
                )
            case "pso":
                return self.__run_pso(kwargs)
            case _:
                raise ValueError(f"Algorithm {algorithm_type} doesn\'t exist")

    def __run_ga(self, **kwargs) -> Solution:
        ga = GAPortfolioOptimization(kwargs)

        return ga.run(seed=kwargs.get('seed'))

    def __run_pso(self, **kwargs) -> Solution:
        return None
