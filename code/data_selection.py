import pandas as pd


def select_data(n: int, path: str):
    data_path = "../database/bodyPerformance.csv"

    data = pd.read_csv(data_path)
    print("Original data shape:", data.shape)

    selected_data = data.sample(n=n, random_state=42)
    print("Selected data shape:", selected_data.shape)

    selected_data.to_csv(path, index=False)
    print("Selected data saved to:", path)



if __name__ == '__main__':
    select_data(n=100, path="../database/data.csv")