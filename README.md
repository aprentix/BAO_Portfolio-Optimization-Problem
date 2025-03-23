Portfolio Optimization Problem 

PG6 
Chen Jin Leonardo, Stefan Oshchypok, Lara Gerlach, Raquel Diaz Chavez 

## Abstract 

A brief summary of the work done in the assignment. It must include: a description of the problem, the methodology followed, key findings, and implications of the work. Limit this to 200-300 words 

The portfolio optimization problem involves finding the best way to allocate assets to achieve a desired combination of risk and return. 

## Problem  

• Background: The Portfolio Optimization Problem consists of finding an allocation of capital among a set of assets to maximize returns while minimizing risk. Whether it’s for managing a multi-billion-dollar institutional portfolio or helping an individual investor grow savings, the principles and methods of portfolio optimization remain central to the field of investment management. Traditionally, methods such as Markowitz’s mean-variance optimization have been popular, but they rely on strong assumptions, such as asset returns being normally distributed. Nevertheless, dealing this problem via Evolutionary Algorithms is a highly attractive alternative, as they can handle non-linear, non-convex, and multi-modal optimization landscapes, as well as relax many of the strict assumptions required by classical methods, and incorporate various real-world constraints like transaction costs, portfolio turnover, and even dynamic market conditions. 

• Problem Definition: Describe the optimization problem in detail, including the constraints, objectives, and mathematical formulations if applicable. 

This real-world problem includes plenty of options and variables, but to make this algorithm more feasible to work with on a smaller scale, we made use of a simpler approach and set the following metrics. 

## Constraints: 

-Budget constraint: Portfolio weights sum to 1 (Σ weights = 1). The budget cannot be exceeded. 

-No short selling: weights ≥ 0. To simplify this problem, we won’t consider selling assets before buying them. 

-Position limits: weight_i ≤ max_allocation [0.05, 0.10]. With the goal of avoiding letting an asset dominate the portfolio, set a hyperparametric value, ensuring diversity (include paper if possible) 

 

## Objectives:  

-Maximizing risk-adjusted returns: Achieved through the calculation of the Sharpe ratio
 

## Procedure:  

Collect your return data over a consistent period (from dataset). 

Determine and align the risk-free rate with your data frequency. 

Compute excess returns for each period. 

Calculate the mean and standard deviation of these excess returns. 

Divide the mean excess return by the standard deviation to get the Sharpe Ratio. 

Annualize, if necessary, by multiplying by the square root of the number of periods per year. 

## Algorithm design  

• Metaheuristics Used: Detail the selected bioinspired metaheuristics for solving the problem. Explain the rationale behind their selection.  

For the Evolutionary Algorithm the chosen metaheuristic was the Genetic Algorithm approach. 

Multi-modal. 

As for the Swam Intelligence Algorithm, the most suitable seemed to be Particle Swarm Optimization approach, because of its real-valued nature. 

The codification chosen for both approaches is a real-valued vector representing the weight of the budget allocated to the different assets. 

• Codification Used: Described the codification used for each metaheuristic in case that they are different. If 

• Implementation Details: Describe the computational tools, programming languages, or software used. Include pseudo-code or algorithms. 

## Experiments Design  

• Parameter Settings: Discuss the configuration of parameters for each metaheuristic that will be tuned, and why/how they have been selected 

• Experimental Setup: Describe how the experiments will be conducted. Specify how the different methods will be compared (what metrics will be used, what statistical tests, etc). Discuss the main limitations of the setup.  

• Data description: Describe the data instances collected to do the experimentation. Describe the sizes, the complexity of each data instance, for what part of the experimentation will be used (parameter tunning, comparison, etc) 


## Experimental results  

• Results: Present the findings from applying the metaheuristics to the optimization problem. Use tables, charts, or graphs for clarity. Analyse the results, highlighting the strengths and weaknesses of each metaheuristic in the context of the problem solved. Discuss any challenges encountered during the experimentation and how they were addressed.  

• Code: Include a link to the repository with all the code needed to replicate the results shown in this section. 

 

## Conclusion and Future Work  

• Summarize the key findings and their implications.  

• Suggest areas for future research or further improvements to the metaheuristics or the problem-solving approach. 


## References  

dataset https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset 

https://www.alphavantage.co/documentation/ 

List all the academic papers, books, and other resources cited in the document. Follow a consistent citation style throughout. 

(use hardvard citation style) 
portfolio optimization problem paper https://link.springer.com/article/10.1007/s00521-024-09456-w 
sharpe ratio https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.1964.tb02865.x 
markowitz portfolio optimization https://math.leidenuniv.nl/scripties/Engels.pdf 
Portfolio diversification Markowitz meets Talmud A combination of sophisticated and naive diversification strategies 

 

-- DATASET --  https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset  

-- DOC -- https://upm365-my.sharepoint.com/:w:/g/personal/leonardo_chenjin_alumnos_upm_es/EcdN9mVXyGNNg-C7yL5i-84BjOZp5b_qnJL5aBGZDyQS1Q?e=UOqNeE  

-- REFERENCE REPO -- https://github.com/naresh-dscience/Portfolio-Optimization-using-Genetic-Algorithm/blob/main/Portfolio_Optimization_Using_GA.ipynb 

 

 

 

 