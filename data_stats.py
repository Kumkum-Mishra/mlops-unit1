import pandas as pd

# Load sample dataset
data = {
    "Age": [22, 25, 30, 35, 40],
    "Salary": [20000, 25000, 30000, 35000, 40000]
}

df = pd.DataFrame(data)

# Print basic statistics
print("Dataset:\n", df)
print("\nStatistics:\n", df.describe())
print("\nMean Salary:", df["Salary"].mean())