import numpy as np


class Solution:
    def __init__(self, weights: list[float], fitness: float):
        self.weights = weights
        self.fitness = fitness

    def decode(self, companies_full_name: list[str], annual_mean_returns: np.array) -> tuple[float, float, dict[str, float]]:
        """
        Converts the list of weights into a tuple containing the solution fitness and a dictionary 
        mapping company names to their assigned weights.
        """
        n_weights = len(self.weights)
        n_companies = len(companies_full_name)
        # Handle mismatch gracefully
        if n_weights != n_companies:
            min_len = min(n_weights, n_companies)
            print(f"[WARNING] Number of weights ({n_weights}) does not match number of companies ({n_companies}). Truncating to {min_len}.")
            weights = self.weights[:min_len]
            companies = companies_full_name[:min_len]
            returns = annual_mean_returns[:min_len]
        else:
            weights = self.weights
            companies = companies_full_name
            returns = annual_mean_returns

        return self.fitness, float(np.sum(np.array(weights) * np.array(returns))), {company: weight for company, weight in zip(companies, weights)}
