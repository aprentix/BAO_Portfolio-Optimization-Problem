import pandas as pd
import os

class DatasetManager:
    DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
    def read_anual_resume(risk_free_rate_annual, start_date, end_date):
        try:
            rfr_folder_name = f"{self.risk_free_rate_annual*100:.1f}".replace(".", "-") + "-risk-free-rate"
            period_folder_name = f"period-from-{self.start_date}-to-{self.end_date}"
            
            annual_resume_companies_path = os.path.join(DATASET_DIR, rfr_folder_name, period_folder_name, "annual_resume_companies.csv")
            df = pd.read_csv(annual_resume_companies_path)
            return df["Mean Excess Return"], df["Volatility"], df["Sharpe Ratio"]
        except Exception as e:
            print(f"[-] Read annual resume error: {e}")
            
    def read_correlation(risk_free_rate_annual, start_date, end_date):
        try:
            rfr_folder_name = f"{self.risk_free_rate_annual*100:.1f}".replace(".", "-") + "-risk-free-rate"
            period_folder_name = f"period-from-{self.start_date}-to-{self.end_date}"
            
            annual_resume_companies_path = os.path.join(DATASET_DIR, rfr_folder_name, period_folder_name, "correlation_companies.csv")
            df = pd.read_csv(annual_resume_companies_path)
            return df
        except Exception as e:
            print(f"[-] Read annual resume error: {e}")
    
    