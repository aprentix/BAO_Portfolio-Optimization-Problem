"""
Portfolio Optimization Problem (POP) package

Portfolio optimization is a fundamental challenge in investment management, focusing on allocating capital among assets to maximize returns while minimizing risk.
This problem is crucial for both institutional investors managing billions of dollars and individuals growing their savings.

"""
__version__ = "1.0.0"

from pop.portfolio_optimization import PortfolioOptimization
from pop.runner import runner


__all__ = [
    'PortfolioOptimization',
    'runner'
]