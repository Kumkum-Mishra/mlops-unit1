import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
# Load dataset
iris = load_iris()
# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
print("First 5 rows:\n", df.head())
# Features and target
X = df.drop("species", axis=1)
y = df["species"]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)
# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# Save model
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as iris_model.pkl")