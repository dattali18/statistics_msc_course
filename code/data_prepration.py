"""
In this code we will further prepare the data for analysis.

We will take a subset of columns from the original dataset, we will remove any string and replace with 0, 1 (male or female).
"""

import pandas as pd

def prepare_data(input_file: str, output_file: str):
    df = pd.read_csv(input_file)

    cols = ['age', 'gender', 'height_cm', 'weight_kg', 'body fat_%']

    renamed_cols = ['age', 'gender', 'height', 'weight', 'bf']

    df = df[cols]

    df.columns = renamed_cols

    # change the 'gender' column from F, M to 0, 1
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})

    # save the data
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = "../database/data.csv"
    output_file = "../database/processed_data.csv"
    prepare_data(input_file, output_file)