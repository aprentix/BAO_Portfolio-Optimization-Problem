from dataset import DatasetManager
from benchmarks import PortfolioOptimization


def main():
    n_companies = 5
    risk_free_rate_annual = 0.042
    start_date = "2015-01-01"
    end_date = "2020-01-01"

    dataset_manager = DatasetManager('dataset')

    _, _, sharpe_ratios, meta = dataset_manager.read_annual_resume(
        risk_free_rate_annual, start_date, end_date, n_companies)

    problem = PortfolioOptimization(
        num_companies=n_companies, sharpe_ratios=sharpe_ratios)

    solution = problem.optimize(algorithm_type="ga")

    print(solution)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
