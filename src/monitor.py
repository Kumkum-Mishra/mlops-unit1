import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Setup logging
logging.basicConfig(
    filename="model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting model training...")

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

logging.info("Model training completed")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model accuracy: {accuracy}")
print("Accuracy:", accuracy)

# Alert system
if accuracy < 0.8:
    logging.warning("Accuracy is low!")
    print("ALERT: Model accuracy is low!")
else:
    logging.info("Model performance is good")