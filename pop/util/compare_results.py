from scipy.stats import wilcoxon, ttest_rel

def perform_statistical_tests(ga_data, pso_data, metric, description, alpha=0.05):
    print(f"\n{description} - Statistical Tests")
    ga_data = ga_data[metric].dropna()
    pso_data = pso_data[metric].dropna()
    if ga_data.empty or pso_data.empty:
        print("⚠️ One of the data sets is empty after removing NaN values. Skipping test.")
        return
    print(f"GA Mean: {ga_data.mean():.4f}, PSO Mean: {pso_data.mean():.4f}")
    try:
        stat, p_value = wilcoxon(ga_data, pso_data)
        print(f"Wilcoxon Test: Statistic={stat:.4f}, P-Value={p_value:.4f}")
        if p_value < alpha:
            print("Difference is statistically significant.")
        else:
            print("Difference is NOT statistically significant.")
    except ValueError as e:
        print(f"Error in Wilcoxon Test: {e}")
    try:
        stat, p_value = ttest_rel(ga_data, pso_data)
        print(f"Paired t-Test: Statistic={stat:.4f}, P-Value={p_value:.4f}")
        if p_value < alpha:
            print("Difference is statistically significant.")
        else:
            print("Difference is NOT statistically significant.")
    except ValueError as e:
        print(f"Error in Paired t-Test: {e}")