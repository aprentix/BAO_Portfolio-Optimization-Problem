# Portfolio Optimization Using Evolutionary Algorithms

## Made by Team `PG6`
- Chen Jin Leonardo
- Stefan Oshchypok
- Lara Gerlach
- Raquel Diaz Chavez

## How to run the project
Use the project management script to set up and execute algorithms. 
Walkthrough:
1. Setup the Project
```bash
python3 manager --setup
```

Other script's commands in help:
```bash
python3 manager -h
```

2. Optimize portfolio
```bash
python3 pop -t ga -n 25
```

3. To see optimization options:
```bash
python3 pop --help
```

## Background

Portfolio optimization is a fundamental challenge in investment management, focusing on allocating capital among assets to maximize returns while minimizing risk. This problem is crucial for both institutional investors managing billions of dollars and individuals growing their savings.

Traditional approaches like Markowitz's mean-variance optimization rely on restrictive assumptions, such as normally distributed asset returns. Evolutionary Algorithms provide a compelling alternative, offering several advantages:

- Ability to handle non-linear, non-convex, and multi-modal optimization landscapes
- Relaxation of strict assumptions required by classical methods
- Incorporation of real-world constraints (transaction costs, portfolio turnover, market dynamics)

## Problem Definition

Portfolio optimization involves numerous variables, but our implementation focuses on a simplified yet effective approach.

### Constraints

1. **Budget Constraint**: Portfolio weights must sum to 1 (Σ weights = 1), ensuring the budget cannot be exceeded
2. **No Short Selling**: All weights must be non-negative (weights ≥ 0), eliminating the complexity of selling assets before purchase
3. **Position Limits**: Individual asset weights cannot exceed maximum allocation thresholds [0.05, 0.10], promoting portfolio diversity and preventing single-asset dominance

### Objective

- **Maximize Risk-Adjusted Returns**: Measured through the Sharpe ratio, which quantifies the relationship between returns and volatility

## Resources

- **Dataset**: [Stock Market Dataset (Kaggle)](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- **Reference Implementation**: [Portfolio Optimization using Genetic Algorithm (GitHub)](https://github.com/naresh-dscience/Portfolio-Optimization-using-Genetic-Algorithm/blob/main/Portfolio_Optimization_Using_GA.ipynb)
- **Examples**: [Teacher repo](https://github.com/panizolledotangel/bao_zubora_gabora)