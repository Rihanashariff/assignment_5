import pandas as pd
import numpy as np
def load_data():
    df1 = pd.read_csv("data/Audible_Catlog.csv")
    df2 = pd.read_csv("data/Audible_Catlog_Advanced_Features.csv")
    return df1, df2

def clean_data(df):
    df.drop_duplicates(inplace=True)
    df.fillna("", inplace=True)
    return df

def merge_data(df1, df2):
    df = pd.merge(df1, df2, on=["Book Name", "Author"], how="left")
    return df

def preprocess():
    df1, df2 = load_data()
    df1 = clean_data(df1)
    df2 = clean_data(df2)
    df = merge_data(df1, df2)
    
    df.to_csv("data/processed_data.csv", index=False)
    print("Data preprocessing completed!")

if __name__ == "__main__":
    preprocess()