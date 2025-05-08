# Experiment Configurations

This folder contains configuration files for Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) experiments.

## File Descriptions

- `ga`: Contains configurations for GA experiments.
- `pso`: Contains configurations for PSO experiments.

## Configuration Fields

### GA Configurations
- `num_assets`: Number of assets
- `correlation_level`: Correlation level ["low", "medium", "high", "None"]
- `pop_size`: Population size.
- `max_generations`: Maximum number of generations.
- `mutation_rate`: Mutation rate.
- `repair_method`: Method to repair invalid solutions.
- `tournament_size`: Size of the tournament for selection.
- `num_elites`: Number of elite individuals to retain.

### PSO Configurations
- `num_assets`: Number of assets
- `correlation_level`: Correlation level ["low", "medium", "high", "None"]
- `pop_size`: Population size.
- `max_iterations`: Maximum number of iterations.
- `inertia_weight`: Inertia weight for velocity updates.
- `cognitive_rate`: Cognitive coefficient.
- `social_rate`: Social coefficient.
- `repair_method`: Method to repair invalid solutions.