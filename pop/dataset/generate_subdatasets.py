from pathlib import Path
import pandas as pd
import os

# Constants
LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.7
NUM_COMPANIES_LIST = [10, 20, 50, 100, 200]

class ExperimentGenerator:
    def __init__(self, base_dir: str):
        """
        Initialize the ExperimentGenerator with a dynamic base directory.

        Args:
            base_dir (str): The base directory for the project.
        """
        self.base_dir = Path(base_dir).parent.parent  # Adjust to point to the root of the project
        self.dataset_dir = Path(base_dir) / "dataset" / "4-2-risk-free-rate" / "period-from-2015-01-01-to-2020-01-01"
        self.correlation_file = self.dataset_dir / "correlation_companies.csv"
        self.experiments_dir = self.base_dir / "experiments" / "subdatasets"

        # Ensure output directory exists
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, file_name: str):
        """
        Load the dataset.

        Args:
            file_name (str): The name of the dataset file.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        file_path = self.dataset_dir / file_name
        df = pd.read_csv(file_path)

        # Rename the first column to 'Company Symbol'
        df.rename(columns={'Unnamed: 0': 'Company Symbol'}, inplace=True)

        # Drop rows with missing or invalid values
        df.dropna(inplace=True)

        # Select relevant columns
        return df[['Company Symbol', 'Mean Excess Return', 'Volatility', 'Sharpe Ratio']]

    def load_correlation_matrix(self):
        """
        Load the precomputed correlation matrix.

        Returns:
            pd.DataFrame: The correlation matrix in long format.
        """
        # Load the correlation matrix
        correlation_df = pd.read_csv(self.correlation_file, index_col=0)

        # Flatten the matrix into long format
        correlation_long = correlation_df.stack().reset_index()
        correlation_long.columns = ['Company 1', 'Company 2', 'Correlation']

        # Remove self-correlations (diagonal elements)
        correlation_long = correlation_long[correlation_long['Company 1'] != correlation_long['Company 2']]

        return correlation_long

    def filter_by_correlation(self, correlation_df: pd.DataFrame):
        """
        Filter companies into low, medium, and high correlation groups.

        Args:
            correlation_df (pd.DataFrame): The correlation matrix in long format.

        Returns:
            tuple: Lists of company symbols for low, medium, and high correlation groups.
        """
        # Filter companies based on correlation thresholds
        low_correlation = correlation_df[correlation_df['Correlation'] <= LOW_THRESHOLD]
        medium_correlation = correlation_df[
            (correlation_df['Correlation'] > LOW_THRESHOLD) & (correlation_df['Correlation'] <= HIGH_THRESHOLD)
        ]
        high_correlation = correlation_df[correlation_df['Correlation'] > HIGH_THRESHOLD]

        # Extract unique company symbols for each group
        low_correlation_companies = set(low_correlation['Company 1']).union(set(low_correlation['Company 2']))
        medium_correlation_companies = set(medium_correlation['Company 1']).union(set(medium_correlation['Company 2']))
        high_correlation_companies = set(high_correlation['Company 1']).union(set(high_correlation['Company 2']))

        return low_correlation_companies, medium_correlation_companies, high_correlation_companies

    def generate_experiment_csvs(self, correlation_group: str, company_symbols: set, dataset: pd.DataFrame):
        """
        Generate experiment CSV files for a given correlation group.

        Args:
            correlation_group (str): Correlation group name ('low', 'medium', 'high').
            company_symbols (set): Set of company symbols for the correlation group.
            dataset (pd.DataFrame): The dataset containing company data.
        """
        # Filter the dataset for the given company symbols
        filtered_df = dataset[dataset['Company Symbol'].isin(company_symbols)]

        for num_companies in NUM_COMPANIES_LIST:
            # Select the top N companies based on Sharpe Ratio
            selected_companies = filtered_df.nlargest(num_companies, 'Sharpe Ratio')

            # Save to CSV
            output_file = self.experiments_dir / f"{correlation_group}_{num_companies}_companies.csv"
            selected_companies.to_csv(output_file, index=False)

            print(f"Experiment file saved: {output_file}")


# Main execution
if __name__ == "__main__":
    # Initialize the generator with the base directory
    generator = ExperimentGenerator(base_dir=os.getcwd())

    # Load the dataset and correlation matrix
    dataset = generator.load_dataset("annual_resume_companies.csv")
    correlation_matrix = generator.load_correlation_matrix()

    # Filter companies by correlation
    low_companies, medium_companies, high_companies = generator.filter_by_correlation(correlation_matrix)

    # Generate experiment CSVs for each correlation group
    generator.generate_experiment_csvs("low", low_companies, dataset)
    generator.generate_experiment_csvs("medium", medium_companies, dataset)
    generator.generate_experiment_csvs("high", high_companies, dataset)