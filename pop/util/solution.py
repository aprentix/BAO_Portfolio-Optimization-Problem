import numpy as np


class Solution:
    def __init__(self, weights: list[float], fitness: float):
        self.weights = weights
        self.fitness = fitness

    def decode(self, companies_full_name: list[str], annual_mean_returns: np.array) -> tuple[float, float, dict[str, float]]:
        """
        Converts the list of weights into a tuple containing the solution fitness and a dictionary 
        mapping company names to their assigned weights.

        Args:
            companies_full_name: List with the full names of the companies
            annual_mean_returns: Companies annual mean return array

        Returns:
            A tuple containing:
                - Portfolio shape ratio (float)
                - Portfolio annual return (float)
                - Dictionary where keys are company names and values are their weights
        """
        if len(self.weights) != len(companies_full_name):
            raise ValueError(
                f"The number of weights ({len(self.weights)}) does not match the number of companies ({len(companies_full_name)})")
        return self.fitness, np.sum(np.array(self.weights) * annual_mean_returns), {company: weight for company, weight in zip(companies_full_name, self.weights)}
