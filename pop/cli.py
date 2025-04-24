import argparse

__version__ = "0.0.0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="POP project runner",
        epilog="Example: %(prog)s --help"
    )

    parser.add_argument("-v", "--version", action="version",
                        version=f"%(prog)s {__version__}")

    global_group = parser.add_argument_group('Global options')
    global_group.add_argument(
        "-t", "--type", choices=["ga", "pso"], help="Choose optimization algorithm")
    global_group.add_argument(
        "-d", "--dataset", help="Setup dataset folder name in project root path", default="dateset")
    global_group.add_argument("-n", "--num-companies",
                              help="Setup number of companies", default=5)
    global_group.add_argument(
        "-r", "--risk", help="Setup annual risk free rate", default=0.042)
    global_group.add_argument(
        "-sd", "--start-day", help="Setup start day of period", default="2015-01-01")
    global_group.add_argument(
        "-ed", "--end-day", help="Setup end day of period", default="2020-01-01")

    other_group = parser.add_argument_group("Other options")
    other_group.add_argument(
        "-l", "--level", choices=["low", "medium", "high"], help="Setup correlation level between companies", default=None)

    return parser.parse_args()
