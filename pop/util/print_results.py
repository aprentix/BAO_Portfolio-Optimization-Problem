import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def print_results(sharpe_ratio, annual_return, weights_dict):
    """
    Displays portfolio optimization results in an attractive format in the console.
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\nSharpe Ratio: {sharpe_ratio}")
    print(f"Annual Return: {annual_return} ({annual_return*100:.2f}%)")
    print("\nPortfolio Distribution:")
    line_format = "{:<45} {:>10} {:>15}"
    print("-" * 80)
    print(line_format.format("Company", "Weight", "Percentage"))
    print("-" * 80)
    for company, weight in sorted(weights_dict.items(), key=lambda x: x[1], reverse=True):
        print(line_format.format(
            company[:44] + "..." if len(company) > 44 else company,
            f"{weight:.4f}",
            f"{weight*100:.2f}%"
        ))
    print("-" * 80)
    total_companies = len(weights_dict)
    active_positions = sum(1 for w in weights_dict.values() if w > 0.001)
    print(f"\nSummary: {active_positions} active positions out of {total_companies} total companies")
    print("=" * 80 + "\n")

def print_statistics(data, metric):
    print(f"\n📊 Statistics for {metric.capitalize()} across configurations:")
    for quality in ["best", "median", "worst"]:
        ga_stats = data[(data["algorithm"] == "GA") & (data["quality"] == quality)][metric].describe()
        pso_stats = data[(data["algorithm"] == "PSO") & (data["quality"] == quality)][metric].describe()
        print(f"\n🔍 {quality.capitalize()} Configuration:")
        print(f"  GA {metric.capitalize()}:")
        print(ga_stats)
        print(f"  PSO {metric.capitalize()}:")
        print(pso_stats)

def plot_fitness_diversity(
    get_results_path,
    algorithm_type,
    quality,
    correlation_level=None
):
    """
    Plot the evolution of fitness and diversity for a given algorithm and config quality.
    """
    try:
        # Build the path to the correct subfolder
        fitness_path = get_results_path(f"{quality}/aggregated_{quality}_aggregated_fitness.csv", algorithm_type.lower(), correlation_level)
        diversity_path = get_results_path(f"{quality}/aggregated_{quality}_aggregated_diversity.csv", algorithm_type.lower(), correlation_level)
        fitness_data = pd.read_csv(fitness_path)
        diversity_data = pd.read_csv(diversity_path)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Fitness
        axes[0].plot(fitness_data['Generation'], fitness_data['Mean Fitness'],
                     label=f'{algorithm_type.upper()} - {quality.capitalize()}')
        axes[0].set_title(f'Fitness Evolution - {algorithm_type.upper()} ({quality.capitalize()})')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Fitness')
        axes[0].grid(True)
        axes[0].legend()

        # Diversity
        axes[1].plot(diversity_data['Generation'], diversity_data['Mean Diversity'],
                     label=f'{algorithm_type.upper()} - {quality.capitalize()}', color='green')
        axes[1].set_title(f'Diversity Evolution - {algorithm_type.upper()} ({quality.capitalize()})')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Diversity')
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        print(f"✅ Plotted fitness and diversity for {algorithm_type.upper()} - {quality.capitalize()}")

    except Exception as e:
        print(f"❌ Could not plot for {algorithm_type.upper()} - {quality.capitalize()}: {e}")

def compare_best_fitness_diversity(get_results_path, correlation_level=None):
    try:
        # Use the quality-based folder and file naming
        for algo in ["ga", "pso"]:
            fitness_path = get_results_path(f"best/aggregated_best_aggregated_fitness.csv", algo, correlation_level)
            diversity_path = get_results_path(f"best/aggregated_best_aggregated_diversity.csv", algo, correlation_level)
            if algo == "ga":
                ga_fitness = pd.read_csv(fitness_path)
                ga_diversity = pd.read_csv(diversity_path)
            else:
                pso_fitness = pd.read_csv(fitness_path)
                pso_diversity = pd.read_csv(diversity_path)

        # Plot fitness and diversity comparisons
        plt.figure(figsize=(14, 6))

        # Fitness Comparison
        plt.subplot(1, 2, 1)
        plt.plot(ga_fitness['Generation'], ga_fitness['Mean Fitness'], label='GA - Best', color='blue', linestyle='-', marker='o')
        plt.plot(pso_fitness['Generation'], pso_fitness['Mean Fitness'], label='PSO - Best', color='green', linestyle='--', marker='x')
        plt.title('Best Fitness Comparison (GA vs PSO)')
        plt.xlabel('Generation')
        plt.ylabel('Mean Fitness')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Diversity Comparison
        plt.subplot(1, 2, 2)
        plt.plot(ga_diversity['Generation'], ga_diversity['Mean Diversity'], label='GA - Best', color='blue', linestyle='-', marker='o')
        plt.plot(pso_diversity['Generation'], pso_diversity['Mean Diversity'], label='PSO - Best', color='green', linestyle='--', marker='x')
        plt.title('Best Diversity Comparison (GA vs PSO)')
        plt.xlabel('Generation')
        plt.ylabel('Mean Diversity')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"❌ Best fitness or diversity data not found: {e}")
    except Exception as e:
        print(f"❌ Error during comparison plotting: {e}")

def plot_time_per_run(selected_configs):
    plt.figure(figsize=(8, 6))
    for algo in ["GA", "PSO"]:
        algo_configs = selected_configs[selected_configs["algorithm"] == algo]
        plt.bar(algo_configs["quality"], algo_configs["mean_time"], label=algo)
    plt.title("Mean Time per Run by Algorithm and Config Quality")
    plt.ylabel("Mean Time (s)")
    plt.legend()
    plt.show()

def plot_fitness_distribution(selected_configs):
    plt.figure(figsize=(8, 6))
    data = [selected_configs[selected_configs["algorithm"] == algo]["mean_sharpe"].dropna() for algo in ["GA", "PSO"]]
    plt.boxplot(data, positions=[0, 1], widths=0.5)
    plt.xticks([0, 1], ["GA", "PSO"])
    plt.title("Distribution of Mean Sharpe Ratios")
    plt.ylabel("Mean Sharpe Ratio")
    plt.show()

def plot_fitness_vs_time(selected_configs):
    plt.figure(figsize=(8, 6))
    for algo in ["GA", "PSO"]:
        algo_configs = selected_configs[selected_configs["algorithm"] == algo]
        plt.scatter(algo_configs["mean_time"], algo_configs["mean_sharpe"], label=algo)
    plt.xlabel("Mean Time (s)")
    plt.ylabel("Mean Sharpe Ratio")
    plt.title("Fitness vs. Time Tradeoff")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fitness_evolution(fitness_history, title="Fitness Evolution Over Generations", save_path=None):
    """
    Plot the fitness evolution over generations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, label="Fitness Evolution", color="blue")
    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        print(f"Fitness evolution plot saved to {save_path}")
    plt.show()
    plt.clf()

def plot_diversity_evolution(diversity_history, title="Diversity Evolution Over Generations", save_path=None):
    """
    Plot the diversity evolution over generations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(diversity_history, label="Diversity Evolution", color="green")
    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel("Diversity")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        print(f"Diversity evolution plot saved to {save_path}")
    plt.show()
    plt.clf()

def bar_plot(data, metric, title, ylabel):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x="quality", y=metric, hue="algorithm", errorbar=None)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()

def box_plot(data, metric, title, ylabel):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x="algorithm", y=metric, hue="quality")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()

def scatter_plot(data, x_metric, y_metric, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_metric, y=y_metric, hue="algorithm", style="quality")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_best_configs_across_correlation(correlation_levels):
    """
    Collects and visualizes the best configurations for GA and PSO across all correlation levels.
    """
    best_configs = []

    for corr_level, file_path in correlation_levels.items():
        data = pd.read_csv(file_path)
        best_ga = data[(data["algorithm"] == "GA") & (data["quality"] == "best")].copy()
        best_ga["correlation"] = corr_level
        best_pso = data[(data["algorithm"] == "PSO") & (data["quality"] == "best")].copy()
        best_pso["correlation"] = corr_level
        best_configs.append(best_ga)
        best_configs.append(best_pso)

    best_configs_df = pd.concat(best_configs, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=best_configs_df,
        x="correlation",
        y="sharpe_ratio",
        hue="algorithm"
    )
    plt.title("Best Sharpe Ratio by Correlation Level (GA vs PSO)")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Correlation Level")
    plt.legend(title="Algorithm")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=best_configs_df,
        x="correlation",
        y="annual_return",
        hue="algorithm"
    )
    plt.title("Best Annual Return by Correlation Level (GA vs PSO)")
    plt.ylabel("Annual Return")
    plt.xlabel("Correlation Level")
    plt.legend(title="Algorithm")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=best_configs_df,
        x="correlation",
        y="runtime",
        hue="algorithm"
    )
    plt.title("Best Execution Time by Correlation Level (GA vs PSO)")
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Correlation Level")
    plt.legend(title="Algorithm")
    plt.show()