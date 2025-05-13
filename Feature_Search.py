import pandas as pd

# Show all columns when printing
pd.set_option('display.max_columns', None)

df = pd.read_csv(r"C:\Users\speco\OneDrive\Desktop\School\finalProjectProposal\BRUIIoT.csv")

# Print the first 5 rows (all columns visible)
print(df.head(5))
num_features = df.shape[1]
print(f"Number of features (columns) in the dataset: {num_features}")