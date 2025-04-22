import os
import pandas as pd
import kagglehub

def download_raw_dataset():
    path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
    print("Path to dataset files:", path)

def construct_table(path):
    df = pd.read_csv(path)
    df = df[["Date","Adj Close"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.to_period('Y')
    df = df.groupby("Date").mean().reset_index()
    df.to_csv(os.path.join("./data/companies", os.path.basename(path)), index= False)

def construct_anual_dataset_anual(num_companies, ini_dataset):
    for dir, sub_dir, files in os.walk(ini_dataset):
        for idx in range(num_companies):
            path_out = os.path.join(dir,files[idx])
            construct_table(path_out)

def contruct_dataset(processed_anual_datasets_path):
    df = pd.DataFrame()
    for dir, sub_dir, files in os.walk(processed_anual_datasets_path):
        for idx in range(len(files)):
            path_out = os.path.join(dir,files[idx])
            name_enterprise = os.path.basename(path_out)[:-4]
            df_temp = pd.read_csv(path_out)
            df_temp['Enterprise'] = name_enterprise
            df_aux = df_temp.pivot(index='Enterprise', columns='Date', values='Adj Close')
            df_aux.reset_index(inplace=True)
            df = pd.concat([df, df_aux], ignore_index=True)
            print(path_out)
    df.to_csv('./data/tabla.csv', index=False)
    return df

# companies = 4
# construct_anual_dataset_anual(companies, './dataset/stocks')
contruct_dataset("./data/companies")