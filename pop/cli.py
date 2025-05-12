import argparse
from datetime import datetime
from pop import __version__

def _valid_date(s):
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{s}'. Valid format: YYYY-MM-DD.")

def parse_args():
    parser = argparse.ArgumentParser(
        prog="pop",
        description="Portfolio Optimization Project (POP) runner",
        epilog="Example: %(prog)s -t ga -n 30 --seed 42",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-v", "--version", action="version",
                        version=f"%(prog)s {__version__}")
    
    # Global options group
    global_group = parser.add_argument_group('Global options')
    global_group.add_argument(
        "-t", "--type", choices=["ga", "pso"], required=True, 
        help="Choose optimization algorithm: Genetic Algorithm (ga) or Particle Swarm Optimization (pso)")
    global_group.add_argument(
        "-d", "--dataset", help="Setup dataset folder name in project root path", 
        default="dataset")
    global_group.add_argument(
        "-n", "--num-companies", type=int, required=True,
        help="Setup number of companies to include in portfolio optimization")
    global_group.add_argument(
        "-r", "--risk", type=float, help="Setup annual risk free rate", 
        default=0.042)
    global_group.add_argument(
        "-sd", "--start-day", type=_valid_date, 
        help="Setup start day of analysis period", default="2015-01-01")
    global_group.add_argument(
        "-ed", "--end-day", type=_valid_date, 
        help="Setup end day of analysis period", default="2020-01-01")
    global_group.add_argument(
        "--seed", type=int, help="Random seed for reproducibility", 
        default=None)
    global_group.add_argument(
        "--save-results", action="store_true", 
        help="Save optimization results to a CSV file")
    global_group.add_argument(
        "--save-fitness", action="store_true", 
        help="Save fitness evolution to a CSV file")
    global_group.add_argument(
        "--save-diversity", action="store_true", 
        help="Save diversity evolution to a CSV file")
    
    # GA specific options
    ga_group = parser.add_argument_group('Genetic Algorithm options (only used when -t ga)')
    ga_group.add_argument(
        "--pop-size", type=int, help="Population size for GA", 
        default=100)
    ga_group.add_argument(
        "--max-generations", type=int, help="Maximum number of generations for GA", 
        default=300)
    ga_group.add_argument(
        "--mutation-rate", type=float, help="Mutation rate for GA", 
        default=0.1)
    ga_group.add_argument(
        "--gaussian-stdev", type=float, help="Standard deviation for Gaussian mutation in GA", 
        default=0.1)
    ga_group.add_argument(
        "--num-elites", type=int, help="Number of elite individuals preserved in each generation for GA", 
        default=1)
    ga_group.add_argument(
        "--tournament-size", type=int, help="Tournament size for selection in GA", 
        default=3)
    ga_group.add_argument(
        "--crossover-prob", type=float, help="Probability of crossover in GA", 
        default=0.9)
    
    # PSO specific options
    pso_group = parser.add_argument_group('Particle Swarm Optimization options (only used when -t pso)')
    pso_group.add_argument(
        "--swarm-size", type=int, help="Swarm size for PSO", 
        default=100)
    pso_group.add_argument(
        "--max-iterations", type=int, help="Maximum number of iterations for PSO", 
        default=300)
    pso_group.add_argument(
        "--w", type=float, help="Inertia weight for PSO", 
        default=0.5)
    pso_group.add_argument(
        "--c1", type=float, help="Cognitive constant for PSO", 
        default=1.5)
    pso_group.add_argument(
        "--c2", type=float, help="Social constant for PSO", 
        default=2.0)
    
    # Other general options
    other_group = parser.add_argument_group("Other options")
    other_group.add_argument(
        "-l", "--level", choices=["low", "medium", "high"], 
        help="Setup correlation level between companies", default=None)
    other_group.add_argument(
        "--repair-method", choices=["normalize", "clip", "restart"], 
        default="normalize", 
        help="Choose repair method for constraint handling")
    
    args = parser.parse_args()
    
    # Validate dates
    if datetime.strptime(args.start_day, "%Y-%m-%d") > datetime.strptime(args.end_day, "%Y-%m-%d"):
        parser.error("Start date must be earlier than or equal to end date.")
    
    return args