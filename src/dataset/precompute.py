import pandas
import os
import shutil
import numpy

STOCK_DATA_DIR = 'stock_data/stocks'
DATASET_DIR = 'dataset'
RISK = "-risk-free-rate"

OUTPUT_FILE = 'daily_returns_companies.csv'

RISK_FREE_RATE_ANNUAL = 0.042 # 4.2% annualized 
TRADING_DAYS = 252

START_DATE = "2015-01-01"
END_DATE = "2020-01-01"

def main():
    stock_data_dir = os.path.join(os.getcwd(), STOCK_DATA_DIR)
    dataset_dir = os.path.join(os.getcwd(), DATASET_DIR)
    main_data_frames = []

    if not os.path.exists(stock_data_dir):
        raise RuntimeError("Stock data not exist")

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
        print(f"Folder created {dataset_dir}")
    files = os.listdir(stock_data_dir)
    for idx in range(5):
        file = files[idx]
        if file.endswith(".csv"):
            path_file = os.path.join(stock_data_dir, file)
            data_frame = pandas.read_csv(path_file, parse_dates=["Date"])
            data_frame = data_frame[["Date", "Adj Close"]].copy()
            data_frame = data_frame.set_index("Date")
            data_frame.rename(columns={"Adj Close": file[:-4]}, inplace=True)
            main_data_frames.append(data_frame)
    main_data_frames = pandas.concat(main_data_frames, axis=1)
    main_data_frames = main_data_frames.pct_change()
    main_data_frames.to_csv(os.path.join(dataset_dir, OUTPUT_FILE))
    folder_path = os.path.join(dataset_dir, str(RISK_FREE_RATE_ANNUAL*100).replace(".","-")+RISK)
    os.mkdir(folder_path)
    risk_free_rate_daily = RISK_FREE_RATE_ANNUAL/TRADING_DAYS
    main_data_frames = main_data_frames - risk_free_rate_daily
    main_data_frames.to_csv(os.path.join(folder_path, "daily_excess_return_companies.csv"))
    folder_path = os.path.join(folder_path, 'period-from-'+START_DATE+"-to-"+END_DATE)
    os.mkdir(folder_path)
    main_data_frames = main_data_frames[main_data_frames["Date"]>=START_DATE & main_data_frames["Date"]<=END_DATE].dropna(axis=1)
    mean_excess = main_data_frames.mean() 
    mean_excess.to_csv(os.path.join(folder_path, "mean_daily_excess_return_companies.csv"))
    std_excess = main_data_frames.std() 
    std_excess.to_csv(os.path.join(folder_path, "daily_volatility_companies.csv"))
    sharpe_ratios = (mean_excess / std_excess) * numpy.sqrt(TRADING_DAYS) 
    sharpe_ratios.to_csv(os.path.join(folder_path, "daily_sharpe_ratio_companies.csv"))

    
    statistics_df = pandas.DataFrame({
        "Mean Excess Return": mean_excess,
        "Volatility": std_excess,
        "Sharpe Ratio": sharpe_ratios
    })
    statistics_df.to_csv(os.path.join(folder_path, "anual_resume_companies.csv"))
    correlation_matrix = main_data_frames.corr()
    correlation_matrix.to_csv(os.path.join(folder_path, "correlation_companies.csv"))


            

if __name__ == "__main__":
    main()
