import pandas as pd
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine(as_frame=True)
df = wine.frame.copy()
df['target_name'] = df['target'].map(lambda i: wine.target_names[i])

# Re-order so the target class label is last
cols = [c for c in df.columns if c!='target_name'] + ['target_name']
df = df[cols]

# Save to CSV
df.to_csv("wine_dataset_assignment.csv", index=False)
print("Saved CSV with shape:", df.shape)
