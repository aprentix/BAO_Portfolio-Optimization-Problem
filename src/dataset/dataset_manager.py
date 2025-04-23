import pandas as pd
import numpy as np
import os
from typing import Optional
from pathlib import Path


class DatasetManager:
    def __init__(self, dir_name: str):
        self.dataset_dir = os.path.join(os.getcwd(), dir_name)

    def read_annual_resume(self, risk_free_rate_annual: float, start_date: str, end_date: str, n_companies: Optional[int] = None, **kwargs):
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

    def read_correlation(self, risk_free_rate_annual: float, start_date: str, end_date: str, n_companies: Optional[int] = None):
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

    def read_annual_resume_same_level_correlation(self, level: str, risk_free_rate_annual: float, start_date: str, end_date: str, n_companies: Optional[int] = None):
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
        full_path = Path(path)
        if not full_path.exists():
            raise FileNotFoundError(f'Path {full_path} is empty')

    def __is_valid_level(self, level: str):
        return level in ["low", "medium", "high"]

    def __get_companies_same_correlation_level(self, avg_corr_per_stock: pd.DataFrame, level: str):
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
