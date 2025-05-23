"""
DatasetManager Module

This module provides functionality for managing financial datasets, particularly for analyzing
stock returns, volatility, Sharpe ratios, and correlations across different time periods
and risk-free rates.

Classes:
    DatasetManager: A class for handling financial datasets stored in specific directory structures.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional
from pathlib import Path


class DatasetManager:
    """
    A class for managing and retrieving financial datasets.

    This class provides methods to read and analyze financial data organized in a specific
    directory structure based on risk-free rates and time periods. It can retrieve annual
    summaries and correlation matrices for company data.

    Attributes:
        dataset_dir (str): The directory path where the datasets are stored.
    """

    def __init__(self, dir_name: str):
        """
        Initialize the DatasetManager with a directory name.

        Args:
            dir_name (str): The name of the directory containing the datasets,
                           relative to the current working directory.
        """
        self._dataset_dir = os.path.join(os.getcwd(), dir_name)

    def read_annual_resume(self, risk_free_rate_annual: float, start_date: str, end_date: str,
                           n_companies: Optional[int] = None, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Read the annual resume data for companies within a specified period and risk-free rate.

        Args:
            risk_free_rate_annual (float): The annual risk-free rate (as a decimal, e.g., 0.05 for 5%).
            start_date (str): The start date of the period in the format expected by the directory structure.
            end_date (str): The end date of the period in the format expected by the directory structure.
            n_companies (Optional[int], optional): Number of companies to limit the results to. Defaults to None.
            **kwargs: Additional keyword arguments.
                companies (list, optional): A list of company identifiers to filter the results.

        Returns:
            tuple: A tuple containing:
                - Mean Excess Return (pandas.Series): Mean excess returns for each company.
                - Volatility (pandas.Series): Volatility measures for each company.
                - Sharpe Ratio (pandas.Series): Sharpe ratios for each company.
                - Companies (list): List of company identifiers.

        Raises:
            FileNotFoundError: If any of the required paths do not exist.
            Exception: For any other errors that occur during reading.
        """
        try:
            base_data_path = self.__get_base_data_path(
                risk_free_rate_annual, start_date, end_date)

            annual_resume_companies_path = os.path.join(
                base_data_path, "annual_resume_companies.csv")

            self.__is_valid_path(annual_resume_companies_path)

            df = pd.read_csv(annual_resume_companies_path, index_col=0)

            companies = kwargs.get('companies')
            if companies is not None:
                df = df.loc[companies]

            if n_companies is not None:
                df = df.head(n_companies)

            return df["Mean Excess Return"], df["Volatility"], df["Sharpe Ratio"], df.index.to_list()
        except Exception as e:
            raise RuntimeError(f"[-] Read annual resume error: {e}")

    def read_correlation(self, risk_free_rate_annual: float, start_date: str, end_date: str,
                         n_companies: Optional[int] = None) -> tuple[pd.DataFrame, list[str]]:
        """
        Read the correlation matrix for companies within a specified period and risk-free rate.

        Args:
            risk_free_rate_annual (float): The annual risk-free rate (as a decimal, e.g., 0.05 for 5%).
            start_date (str): The start date of the period in the format expected by the directory structure.
            end_date (str): The end date of the period in the format expected by the directory structure.
            n_companies (Optional[int], optional): Number of companies to limit the correlation matrix to.
                                                Defaults to None.

        Returns:
            tuple: A tuple containing:
                - Correlation Matrix (pandas.DataFrame): The correlation matrix between companies.
                - Companies (list): List of company identifiers.

        Raises:
            FileNotFoundError: If any of the required paths do not exist.
            Exception: For any other errors that occur during reading.
        """
        try:
            base_data_path = self.__get_base_data_path(
                risk_free_rate_annual, start_date, end_date)

            correlation_companies_path = os.path.join(
                base_data_path, "correlation_companies.csv")

            self.__is_valid_path(correlation_companies_path)

            df = pd.read_csv(correlation_companies_path, index_col=0)

            return df if n_companies is None else df.iloc[0:n_companies, 0:n_companies], df.index.to_list()
        except Exception as e:
            raise RuntimeError(f"[-] Read correlation error: {e}")

    def read_annual_resume_same_level_correlation(self, level: str, risk_free_rate_annual: float,
                                                  start_date: str, end_date: str,
                                                  n_companies: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Read annual resume data filtered by correlation level.

        Args:
            level (str): The correlation level to filter by. Must be one of: "low", "medium", "high".
            risk_free_rate_annual (float): The annual risk-free rate (as a decimal, e.g., 0.05 for 5%).
            start_date (str): The start date of the period in the format expected by the directory structure.
            end_date (str): The end date of the period in the format expected by the directory structure.
            n_companies (Optional[int], optional): Number of companies to limit the results to. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - Mean Excess Return (pandas.Series): Mean excess returns for filtered companies.
                - Volatility (pandas.Series): Volatility measures for filtered companies.
                - Sharpe Ratio (pandas.Series): Sharpe ratios for filtered companies.
                - Companies (list): List of filtered company identifiers.
        """
        if not self.__is_valid_level(level):
            raise ValueError(f"This level of correlation doesn't exist: {level}")

        # Step 1: Read the correlation matrix
        corr_df, _ = self.read_correlation(risk_free_rate_annual, start_date, end_date)

        # Step 2: Filter companies based on the correlation level
        filter_companies = self.__get_companies_same_correlation_level(corr_df, level)

        # Step 3: Read the annual resume for the filtered companies
        return self.read_annual_resume(risk_free_rate_annual, start_date, end_date, n_companies=n_companies, companies=filter_companies)

    def get_full_companies_names(self, symbols):
        """
        Retrieve full company information for a list of ticker symbols.

        This method looks up company details from a metadata CSV file using ticker symbols.

        Args:
            symbols (list): A list of ticker symbols to look up.

        Returns:
            pd.DataFrame: A DataFrame containing all available metadata for the requested symbols.

        Raises:
            FileNotFoundError: If the symbols metadata file does not exist.
            KeyError: If any of the requested symbols are not found in the metadata.
        """
        symbols_path = os.path.join(self._dataset_dir, "symbols_valid_meta.csv")
        self.__is_valid_path(symbols_path)
        df = pd.read_csv(symbols_path, index_col=1)

        # Only keep symbols that exist in the metadata
        available = [s for s in symbols if s in df.index]
        missing = [s for s in symbols if s not in df.index]

        if missing:
            print(f"[WARNING] The following symbols are missing from metadata and will be skipped: {missing}")

        if not available:
            raise KeyError("None of the requested symbols were found in metadata.")

        return df.loc[available, "Security Name"]

    def __get_base_data_path(self, risk_free_rate_annual: float, start_date: str, end_date: str) -> str:
        """
        Construct and validate the base path for accessing data for a specific risk-free rate and time period.

        This private method builds the directory path based on the risk-free rate and date range,
        validates that the path exists, and returns the complete path for further use.

        Args:
            risk_free_rate_annual (float): The annual risk-free rate (as a decimal, e.g., 0.05 for 5%).
            start_date (str): The start date of the period in the format expected by the directory structure.
            end_date (str): The end date of the period in the format expected by the directory structure.

        Returns:
            str: The full path to the directory containing data for the specified parameters.

        Raises:
            FileNotFoundError: If either the risk-free rate directory or the period directory does not exist.
        """
        rfr_folder_name = f"{risk_free_rate_annual*100:.1f}".replace(
            ".", "-") + "-risk-free-rate"

        rfr_folder_path = os.path.join(self._dataset_dir, rfr_folder_name)

        self.__is_valid_path(rfr_folder_path)

        period_folder_name = f"period-from-{start_date}-to-{end_date}"

        period_folder_path = os.path.join(rfr_folder_path, period_folder_name)

        self.__is_valid_path(period_folder_path)

        return period_folder_path

    def __is_valid_path(self, path: str):
        """
        Check if a path exists. If not, raise a FileNotFoundError.

        Args:
            path (str): The path to validate.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        full_path = Path(path)
        if not full_path.exists():
            raise FileNotFoundError(f'Path {full_path} is empty')

    def __is_valid_level(self, level: str):
        """
        Check if a correlation level is valid.

        Args:
            level (str): The correlation level to validate.

        Returns:
            bool: True if the level is valid, False otherwise.
        """
        return level in ["low", "medium", "high"]

    def __get_companies_same_correlation_level(self, corr_matrix: pd.DataFrame, level: str) -> list:
        """
        Filter companies based on their pairwise Pearson correlation level.

        Args:
            corr_matrix (pd.DataFrame): Correlation matrix of companies.
            level (str): The correlation level to filter by. Must be one of: "low", "medium", "high".

        Returns:
            list: A list of company identifiers that match the specified correlation level.

        Raises:
            ValueError: If the specified correlation level doesn't exist.
        """
        if not self.__is_valid_level(level):
            raise ValueError(f"Invalid correlation level: {level}. Must be 'low', 'medium', or 'high'.")

        # Step 1: Exclude diagonal values (self-correlation)
        np.fill_diagonal(corr_matrix.values, np.nan)

        # Step 2: Flatten the matrix to get all pairwise correlations
        pairwise_corr = corr_matrix.unstack().dropna()

        # Step 3: Determine dynamic thresholds using quantiles
        low_threshold = pairwise_corr.quantile(0.33)
        high_threshold = pairwise_corr.quantile(0.66)

        # Step 4: Filter companies based on the specified level
        if level == "low":
            filtered_pairs = pairwise_corr[pairwise_corr < low_threshold]
        elif level == "medium":
            filtered_pairs = pairwise_corr[(pairwise_corr >= low_threshold) & (pairwise_corr <= high_threshold)]
        elif level == "high":
            filtered_pairs = pairwise_corr[pairwise_corr > high_threshold]

        # Step 5: Extract unique company identifiers from the filtered pairs
        filtered_companies = set(filtered_pairs.index.get_level_values(0)).union(
            set(filtered_pairs.index.get_level_values(1))
        )

        return list(filtered_companies)
