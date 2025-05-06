import time
from experiments.hyperparameters.tuning_loader import load_best_config
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon, ttest_ind
from pop.runner import runner
from matplotlib.backends.backend_pdf import PdfPages

# Configuration
NUM_RUNS = 30
NUM_COMPANIES = 50
START_DATE = "2015-01-01"
END_DATE = "2020-01-01"
RISK_FREE_RATE = 0.042
DATASET = "dataset"

REPAIR_METHODS = ["normalize", "clip", "restart", "shrink"]

# GA parameters
ga_params_base = {
    "pop_size": 100,
    "max_generations": 300,
    "mutation_rate": 0.1,
    "gaussian_stdev": 0.1,
    "num_elites": 2
}

# PSO parameters
pso_params_base = {
    "pop_size": 100,
    "max_generations": 300,
    "w": 0.6,
    "c1": 1.5,
    "c2": 2.0
}


def run_multiple(algorithm_type: str, repair_method: str, params: dict):
    results, times, returns = [], [], []
    for _ in range(NUM_RUNS):
        start = time.time()
        sharpe, ret, _ = runner(
            algorithm_type=algorithm_type,
            dataset_folder_name=DATASET,
            num_companies=NUM_COMPANIES,
            risk_free_rate_annual=RISK_FREE_RATE,
            start_date=START_DATE,
            end_date=END_DATE,
            repair_method=repair_method,
            **params
        )
        duration = time.time() - start
        results.append(sharpe)
        returns.append(ret)
        times.append(duration)
    return results, returns, times


def plot_all_boxplots(metric_dict, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    labels = list(metric_dict.keys())
    data = [metric_dict[k] for k in labels]
    plt.boxplot(data, tick_labels=labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def perform_stats(metric_dict, metric_name):
    lines = [f"{metric_name} Comparison Across Repair Methods:"]
    for method1 in metric_dict:
        for method2 in metric_dict:
            if method1 < method2:
                t_stat, t_p = ttest_ind(metric_dict[method1], metric_dict[method2])
                w_stat, w_p = wilcoxon(metric_dict[method1], metric_dict[method2])
                lines.append(f"{method1} vs {method2}: t = {t_stat:.4f}, p = {t_p:.4f} | W = {w_stat:.4f}, p = {w_p:.4f}")
    return lines

def main():
    all_results = []

    for repair in REPAIR_METHODS:
        # without fine_tuning.py results:
        #print(f"\nRunning GA with repair: {repair}")
        #ga_scores, ga_returns, ga_times = run_multiple("ga", repair, ga_params_base)

        #print(f"Running PSO with repair: {repair}")
        #pso_scores, pso_returns, pso_times = run_multiple("pso", repair, pso_params_base)

        # with fine_tuning.py results:
        best_ga_params = load_best_config("fine_tuning_results.csv", algorithm="GA")
        best_pso_params = load_best_config("fine_tuning_results.csv", algorithm="PSO")

        ga_scores, ga_returns, ga_times = run_multiple("ga", "normalize", best_ga_params)
        pso_scores, pso_returns, pso_times = run_multiple("pso", "normalize", best_pso_params)

        for s, r, t, alg in zip([ga_scores, pso_scores], [ga_returns, pso_returns], [ga_times, pso_times], ["GA", "PSO"]):
            all_results.append(pd.DataFrame({
                "algorithm": alg,
                "repair_method": repair,
                "sharpe_ratio": s,
                "return": r,
                "runtime": t
            }))

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv("results_repair_methods.csv", index=False)
    print("\nResults saved to results_repair_methods.csv")

    # Export to PDF
    plot_data = {}
    for metric in ["sharpe_ratio", "return", "runtime"]:
        for alg in ["GA", "PSO"]:
            subset = df_all[df_all["algorithm"] == alg]
            metric_dict = {rm: subset[subset["repair_method"] == rm][metric].tolist() for rm in REPAIR_METHODS}
            labels = list(metric_dict.keys())
            data = [metric_dict[k] for k in labels]
            plot_data[f"{metric.title()} - {alg}"] = (data, labels, metric.title())
            perform_stats(metric_dict, f"{metric.title()} ({alg})")

        stats_results = {}
    for metric in ["sharpe_ratio", "return", "runtime"]:
        for alg in ["GA", "PSO"]:
            subset = df_all[df_all["algorithm"] == alg]
            metric_dict = {rm: subset[subset["repair_method"] == rm][metric].tolist() for rm in REPAIR_METHODS}
            labels = list(metric_dict.keys())
            data = [metric_dict[k] for k in labels]
            plot_data[f"{metric.title()} - {alg}"] = (data, labels, metric.title())

            # Save statistics
            stats_results[f"{metric.title()} ({alg})"] = perform_stats(metric_dict, f"{metric.title()} ({alg})")

    export_plots_to_pdf(plot_data)
    export_text_to_pdf(stats_results)


def export_text_to_pdf(text_blocks: dict, filename="stats_summary.pdf"):
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(filename) as pdf:
        for title, lines in text_blocks.items():
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            wrapped_text = "\n".join(lines)
            ax.text(0.01, 0.99, f"{title}\n\n{wrapped_text}", va='top', ha='left', fontsize=10, wrap=True)
            pdf.savefig(fig)
            plt.close()
    print(f" Exported statistics to {filename}")

def export_plots_to_pdf(plot_data: dict, filename="repair_method_comparison.pdf"):
    with PdfPages(filename) as pdf:
        for title, (data, labels, ylabel) in plot_data.items():
            plt.figure(figsize=(10, 6))
            plt.boxplot(data, tick_labels=labels)
            plt.title(title)
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"Exported all plots to {filename}")

if __name__ == "__main__":
    main()
