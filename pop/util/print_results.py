def print_results(sharpe_ratio, annual_return, weights_dict):
    """
    Displays portfolio optimization results in an attractive format in the console.

    Args:
        sharpe_ratio: The Sharpe ratio of the optimized portfolio
        annual_return: The annual return of the optimized portfolio
        weights_dict: Dictionary with weights assigned to each company
    """
    # Print header
    print("\n" + "=" * 80)
    print(" " * 25 + "PORTFOLIO OPTIMIZATION RESULTS")
    print("=" * 80)

    # Main metrics
    print(f"\nSharpe Ratio: {sharpe_ratio}")
    print(f"Annual Return: {annual_return} ({annual_return*100}%)")

    # Portfolio distribution
    print("\nPortfolio Distribution:")

    # Create a line format for consistency
    line_format = "{:<45} {:>10} {:>15}"

    # Print the header for the table
    print("-" * 80)
    print(line_format.format("Company", "Weight", "Percentage"))
    print("-" * 80)

    # Sort the weights by value (descending) and print each company
    for company, weight in sorted(weights_dict.items(), key=lambda x: x[1], reverse=True):
        print(line_format.format(
            company[:44] + "..." if len(company) > 44 else company,
            f"{weight:.4f}",
            f"{weight*100:.2f}%"
        ))

    print("-" * 80)

    # Summary information
    total_companies = len(weights_dict)
    # Significant weight positions
    active_positions = sum(1 for w in weights_dict.values() if w > 0.001)

    print(
        f"\nSummary: {active_positions} active positions out of {total_companies} total companies")
    print("=" * 80 + "\n")
