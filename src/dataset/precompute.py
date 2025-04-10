#!/usr/bin/env python3
"""
Stock Data Analysis Tool

This script analyzes stock data to calculate various financial metrics including:
- Daily returns
- Excess returns
- Volatility
- Sharpe ratios
- Correlation matrices

The script reads stock data from CSV files and outputs analysis results to specified directories.
"""

import os
import shutil
from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path


class StockAnalyzer:
    """Class for analyzing stock data and calculating financial metrics."""

    def __init__(self,
                 stock_data_root: str,
                 stock_data_dir: str,
                 output_dir: str,
                 risk_free_rate_annual: float = 0.042,
                 trading_days: int = 252,
                 start_date: str = "2015-01-01",
                 end_date: str = "2020-01-01",
                 max_stocks: Optional[int] = None):
        """
        Initialize the StockAnalyzer with paths and date ranges.

        Args:
            stock_data_dir: Directory containing stock CSV files
            output_dir: Directory for output analysis files
            start_date: Analysis start date in YYYY-MM-DD format
            end_date: Analysis end date in YYYY-MM-DD format
            max_stocks: Maximum number of stocks to analyze (None for all)
        """
        self.stock_data_root = Path(stock_data_root)
        self.stock_data_dir = Path(stock_data_dir)
        self.output_dir = Path(output_dir)
        self.risk_free_rate_annual = risk_free_rate_annual
        self.trading_days = trading_days
        self.start_date = start_date
        self.end_date = end_date
        self.max_stocks = max_stocks

        # Output file names
        self.file_names = {
            'daily_returns': 'daily_returns_companies.csv',
            'daily_excess_returns': 'daily_excess_return_companies.csv',
            'mean_daily_excess': 'mean_daily_excess_return_companies.csv',
            'daily_volatility': 'daily_volatility_companies.csv',
            'sharpe_ratio': 'daily_sharpe_ratio_companies.csv',
            'annual_resume': 'annual_resume_companies.csv',
            'correlation': 'correlation_companies.csv',
            'symbols_valid_meta':  'symbols_valid_meta.csv'
        }

        # Calculated risk-free rate (daily)
        self.risk_free_rate_daily = self.risk_free_rate_annual / self.trading_days

        # Create needed directories
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary output directories if they don't exist."""
        # Check if stock data directory exists
        if not self.stock_data_dir.exists():
            raise FileNotFoundError(
                f"Stock data directory not found: {self.stock_data_dir}")

        # Create main output directory
        self.output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Create risk-free rate directory
        rfr_folder_name = f"{self.risk_free_rate_annual*100:.1f}".replace(
            ".", "-") + "-risk-free-rate"
        self.rfr_dir = self.output_dir / rfr_folder_name
        self.rfr_dir.mkdir(exist_ok=True)
        print(f"Risk-free rate directory: {self.rfr_dir}")

        # Create period-specific directory
        period_folder_name = f"period-from-{self.start_date}-to-{self.end_date}"
        self.period_dir = self.rfr_dir / period_folder_name
        self.period_dir.mkdir(exist_ok=True)
        print(f"Period directory: {self.period_dir}")

    def copy_symbols_file(self) -> None:
        """Copy the symbols metadata file to the output directory."""
        try:
            source_file = self.stock_data_root / \
                self.file_names['symbols_valid_meta']
            target_file = self.output_dir / \
                self.file_names['symbols_valid_meta']
            if source_file.exists():
                shutil.copy(source_file, target_file)
                print(f"Copied symbols file to {target_file}")
            else:
                print(f"Warning: Symbols file not found at {source_file}")
        except Exception as e:
            print(f"Error copying symbols file: {e}")

    def load_stock_data(self) -> pd.DataFrame:
        """
        Load adjusted close prices from stock CSV files.

        Returns:
            DataFrame containing adjusted close prices for all stocks
        """
        adj_close_data_frames = []

        # Get all CSV files in stock directory
        csv_files = [f for f in self.stock_data_dir.glob("*.csv")
                     if f.name != "symbols_valid_meta.csv"]

        # Limit number of stocks if specified
        if self.max_stocks is not None:
            csv_files = csv_files[:self.max_stocks]

        print(f"Loading data for {len(csv_files)} stocks...")

        for file_path in csv_files:
            try:
                # Read and process each stock file
                data_frame = pd.read_csv(file_path, parse_dates=["Date"])
                data_frame = data_frame[["Date", "Adj Close"]].copy()
                data_frame = data_frame.set_index("Date")

                # Use file name without extension as column name
                stock_symbol = file_path.stem
                data_frame.rename(
                    columns={"Adj Close": stock_symbol}, inplace=True)

                adj_close_data_frames.append(data_frame)
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

        # Combine all stock data
        if not adj_close_data_frames:
            raise ValueError("No valid stock data files found")

        returns_data_frames = pd.concat(adj_close_data_frames, axis=1)
        return returns_data_frames

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from price data.

        Args:
            prices: DataFrame with stock price data

        Returns:
            DataFrame with daily returns
        """
        returns = prices.pct_change()
        return returns

    def calculate_excess_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate excess returns by subtracting risk-free rate.

        Args:
            returns: DataFrame with daily returns

        Returns:
            DataFrame with excess returns
        """
        excess_returns = returns.subtract(self.risk_free_rate_daily)
        return excess_returns

    def filter_by_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to specified date range and remove NaN values.

        Args:
            df: DataFrame to filter

        Returns:
            Filtered DataFrame
        """
        filtered_df = df.loc[self.start_date:self.end_date].dropna(axis=1)
        return filtered_df

    def calculate_statistics(self, mean_excess: pd.DataFrame, std_excess: pd.DataFrame, sharpe_ratios: pd.DataFrame) -> pd.DataFrame:
        """
        Combine pre-calculated statistics into a annual summary DataFrame.

        Args:
            mean_excess: Series with mean excess returns (daily)
            std_excess: Series with standard deviations (daily)
            sharpe_ratios: Series with Sharpe ratios (daily)

        Returns:
            DataFrame with combined statistics
        """
        annual_mean_excess = mean_excess * self.trading_days
        annual_std_excess = std_excess * np.sqrt(self.trading_days)
        annual_sharpe_ratios = sharpe_ratios * np.sqrt(self.trading_days)

        statistics_df = pd.DataFrame({
            "Mean Excess Return": annual_mean_excess,
            "Volatility": annual_std_excess,
            "Sharpe Ratio": annual_sharpe_ratios
        })

        return statistics_df

    def save_dataframe(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Save DataFrame to CSV with proper error handling.

        Args:
            df: DataFrame to save
            output_path: Path to save the CSV file
        """
        try:
            df.to_csv(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error saving {output_path}: {e}")

    def run_analysis(self) -> None:
        """Run the complete stock analysis pipeline."""
        # Copy symbols file
        self.copy_symbols_file()

        # Check if daily returns file already exists
        daily_returns_path = self.output_dir / self.file_names['daily_returns']
        daily_excess_returns_path = self.rfr_dir / \
            self.file_names['daily_excess_returns']

        returns = None
        excess_returns = None

        if not daily_returns_path.exists():
            print("Loading stock data and calculating returns...")
            # Load price data
            prices = self.load_stock_data()
            # Calculate daily returns
            returns = self.calculate_returns(prices)
            self.save_dataframe(returns, self.output_dir /
                                self.file_names['daily_returns'])
        else:
            print(f"Using existing returns file {daily_returns_path}:")
            returns = pd.read_csv(daily_returns_path,
                                  index_col=0, parse_dates=True)

        if not daily_excess_returns_path.exists():
            # Calculate excess returns
            excess_returns = self.calculate_excess_returns(returns)
            self.save_dataframe(excess_returns, self.rfr_dir /
                                self.file_names['daily_excess_returns'])
        else:
            print(f"Using existing excess returns file {daily_excess_returns_path}:")
            excess_returns = pd.read_csv(
                daily_excess_returns_path, index_col=0, parse_dates=True)

        # Filter by date range
        filtered_excess_returns = self.filter_by_date_range(excess_returns)

        # Calculate and save mean excess returns
        mean_excess = filtered_excess_returns.mean()
        self.save_dataframe(mean_excess, self.period_dir /
                            self.file_names['mean_daily_excess'])

        # Calculate and save volatility
        std_excess = filtered_excess_returns.std()
        self.save_dataframe(std_excess, self.period_dir /
                            self.file_names['daily_volatility'])

        # Calculate and save Sharpe ratios
        sharpe_ratios = (mean_excess / std_excess)
        self.save_dataframe(sharpe_ratios, self.period_dir /
                            self.file_names['sharpe_ratio'])

        # Calculate and save statistics summary
        statistics_df = self.calculate_statistics(
            mean_excess=mean_excess, std_excess=std_excess, sharpe_ratios=sharpe_ratios)
        self.save_dataframe(statistics_df, self.period_dir /
                            self.file_names['annual_resume'])

        # Calculate and save correlation matrix
        correlation_matrix = filtered_excess_returns.corr()
        self.save_dataframe(correlation_matrix,
                            self.period_dir / self.file_names['correlation'])

        print("Analysis completed successfully!")


def main():
    """Main entry point for the script."""
    # Define paths relative to current working directory
    stock_data_root = os.path.join(os.getcwd(), 'stock_data')
    stock_data_dir = os.path.join(os.getcwd(), 'stock_data/stocks')
    output_dir = os.path.join(os.getcwd(), 'dataset')

    try:
        # Create analyzer with default settings
        analyzer = StockAnalyzer(
            stock_data_root=stock_data_root,
            stock_data_dir=stock_data_dir,
            output_dir=output_dir,
            risk_free_rate_annual=0.042,
            trading_days=252,
            start_date="2015-01-01",
            end_date="2020-01-01",
            max_stocks=5
        )

        # Run analysis
        analyzer.run_analysis()

    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
