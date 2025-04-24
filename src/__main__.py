import sys

from .cli import parse_args
from .runner import runner

def main():
    args = parse_args()

    return runner(args.dataset, args.num_companies, args.risk, args.start_day, args.end_day, correlation_level=args.level)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
        sys.exit(1)