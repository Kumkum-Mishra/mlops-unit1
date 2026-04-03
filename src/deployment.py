from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model A (old)
model_a = LogisticRegression(max_iter=200)
model_a.fit(X_train, y_train)
pred_a = model_a.predict(X_test)
acc_a = accuracy_score(y_test, pred_a)

# Model B (new)
model_b = RandomForestClassifier()
model_b.fit(X_train, y_train)
pred_b = model_b.predict(X_test)
acc_b = accuracy_score(y_test, pred_b)

print("Model A Accuracy:", acc_a)
print("Model B Accuracy:", acc_b)

# A/B Testing logic
if acc_b > acc_a:
    print("Deploy Model B (Better)")
else:
    print("Keep Model A")