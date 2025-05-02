import argparse
from datetime import datetime

__version__ = "0.0.0"


def _valid_date(s):
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: '{s}'. Valid format: YYYY-MM-DD.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="POP project runner",
        epilog="Example: %(prog)s --help",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-v", "--version", action="version",
                        version=f"%(prog)s {__version__}")

    global_group = parser.add_argument_group('Global options')
    global_group.add_argument(
        "-t", "--type", choices=["ga", "pso"], required=True, help="Choose optimization algorithm")
    global_group.add_argument(
        "-d", "--dataset", help="Setup dataset folder name in project root path", default="dataset")
    global_group.add_argument("-n", "--num-companies", type=int, required=True,
                              help="Setup number of companies")
    global_group.add_argument(
        "-r", "--risk", type=float, help="Setup annual risk free rate", default=0.042)
    global_group.add_argument(
        "-sd", "--start-day", type=_valid_date, help="Setup start day of period", default="2015-01-01")
    global_group.add_argument(
        "-ed", "--end-day", type=_valid_date, help="Setup end day of period", default="2020-01-01")

    other_group = parser.add_argument_group("Other options")
    other_group.add_argument(
        "-l", "--level", choices=["low", "medium", "high"], help="Setup correlation level between companies", default=None)
    
    other_group.add_argument("--pop-size", type=int, help="Population size", default=100)
    other_group.add_argument("--max-generations", type=int, help="Maximum number of generations or iterations", default=300)
    other_group.add_argument("--mutation-rate", type=float, help="Mutation rate for GA", default=0.1)
    other_group.add_argument("--gaussian-stdev", type=float, help="Standard deviation for Gaussian mutation in GA", default=0.1)
    other_group.add_argument("--num-elites", type=int, help="Number of elite individuals in GA", default=1)
    other_group.add_argument("--w", type=float, help="Inertia weight for PSO", default=0.5)
    other_group.add_argument("--c1", type=float, help="Cognitive constant for PSO", default=1.5)
    other_group.add_argument("--c2", type=float, help="Social constant for PSO", default=2.0)
    other_group.add_argument("--repair-method", choices=["normalize", "clip", "restart"], default="normalize", help="Choose repair method for constraint handling")

    args = parser.parse_args()

    if datetime.strptime(args.start_day, "%Y-%m-%d") > datetime.strptime(args.end_day, "%Y-%m-%d"):
        parser.error("Start date must be earlier than or equal to end date.")

    return args
