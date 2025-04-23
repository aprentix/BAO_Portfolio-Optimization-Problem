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
        self.dataset_dir = os.path.join(os.getcwd(), dir_name)

    def read_annual_resume(self, risk_free_rate_annual: float, start_date: str, end_date: str,
                           n_companies: Optional[int] = None, **kwargs):
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
            rfr_folder_name = f"{risk_free_rate_annual*100:.1f}".replace(
                ".", "-") + "-risk-free-rate"

            self.__is_valid_path(rfr_folder_name)

            period_folder_name = f"period-from-{start_date}-to-{end_date}"

            self.__is_valid_path(period_folder_name)

            annual_resume_companies_path = os.path.join(
                self.dataset_dir, rfr_folder_name, period_folder_name, "annual_resume_companies.csv")

            self.__is_valid_path(annual_resume_companies_path)

            df = pd.read_csv(annual_resume_companies_path, index_col=0)

            companies = kwargs.get('companies')
            if companies is not None:
                df = df.loc[companies]

            if n_companies is not None:
                df = df.head(n_companies)

            return df["Mean Excess Return"], df["Volatility"], df["Sharpe Ratio"], df.index.to_list()
        except Exception as e:
            print(f"[-] Read annual resume error: {e}")

    def read_correlation(self, risk_free_rate_annual: float, start_date: str, end_date: str,
                         n_companies: Optional[int] = None):
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
            rfr_folder_name = f"{risk_free_rate_annual*100:.1f}".replace(
                ".", "-") + "-risk-free-rate"

            self.__is_valid_path(rfr_folder_name)

            period_folder_name = f"period-from-{start_date}-to-{end_date}"

            self.__is_valid_path(period_folder_name)

            correlation_companies_path = os.path.join(
                self.dataset_dir, rfr_folder_name, period_folder_name, "correlation_companies.csv")

            self.__is_valid_path(correlation_companies_path)

            df = pd.read_csv(correlation_companies_path, index_col=0)

            return df if n_companies is None else df.iloc[0:n_companies, 0:n_companies], df.index.to_list()
        except Exception as e:
            print(f"[-] Read correlation error: {e}")

    def read_annual_resume_same_level_correlation(self, level: str, risk_free_rate_annual: float,
                                                  start_date: str, end_date: str,
                                                  n_companies: Optional[int] = None):
        """
        Read annual resume data filtered by correlation level.

        This method filters companies based on their average correlation level (low, medium, high)
        and returns their annual resume data.

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

        Raises:
            ValueError: If the specified correlation level doesn't exist.
            FileNotFoundError: If any of the required paths do not exist.
            Exception: For any other errors that occur during reading.
        """
        if not self.__is_valid_level(level):
            raise ValueError(
                f'This level of correlation doesn\'t exist: {level}')

        corr_df = self.read_correlation(
            risk_free_rate_annual, start_date, end_date)

        avg_corr_per_stock = corr_df.apply(lambda row: (
            np.abs(row.drop(row.name))).mean(), axis=1)

        filter_companies = self.__get_companies_same_correlation_level(
            avg_corr_per_stock, level)

        return self.read_annual_resume(risk_free_rate_annual, start_date, end_date, n_companies=n_companies, companies=filter_companies)

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

    def __get_companies_same_correlation_level(self, avg_corr_per_stock: pd.DataFrame, level: str):
        """
        Filter companies based on their average correlation level.

        Args:
            avg_corr_per_stock (pd.DataFrame): DataFrame containing average correlation values for each stock.
            level (str): The correlation level to filter by. Must be one of: "low", "medium", "high".

        Returns:
            list: A list of company identifiers that match the specified correlation level.

        Raises:
            ValueError: If the specified correlation level doesn't exist.
        """
        LOW_THRESHOLD = 0.333333
        HIGH_THRESHOLD = 0.666666

        match(level):
            case "low":
                return avg_corr_per_stock[avg_corr_per_stock < LOW_THRESHOLD].index.to_list()
            case "medium":
                return avg_corr_per_stock[(avg_corr_per_stock >= LOW_THRESHOLD) & (avg_corr_per_stock <= HIGH_THRESHOLD)].index.to_list()
            case "high":
                return avg_corr_per_stock[avg_corr_per_stock > HIGH_THRESHOLD].index.to_list()
            case _:
                raise ValueError(
                    f'This level of correlation doesn\'t exist: {level}')
