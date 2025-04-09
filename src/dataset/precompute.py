import pandas
import os
import shutil

STOCK_DATA_DIR = 'stock_data'
DATASET_DIR = 'dataset'

RISK_FREE_RATE_ANNUAL = 0.042
TRADING_DAYS = 252

START_DATE = "2015-01-01"
END_DATE = "2020-01-01"

def main():
    stock_data_dir = os.path.join(os.getcwd(), STOCK_DATA_DIR)
    dataset_dir = os.path.join(os.getcwd(), DATASET_DIR)

    if not os.path.exists(stock_data_dir):
        raise Exception("Stock data not exist")

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
        print(f"Folder created {dataset_dir}")

if __name__ == "__main__":
    main()
