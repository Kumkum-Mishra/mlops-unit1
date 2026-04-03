# MLOps Unit 1 Lab

This repository contains all practical exercises for MLOps Unit 1, including Git version control, machine learning workflow, experiment tracking, versioning, CI/CD, deployment strategies, monitoring, and optimization.

---

## 📁 Project Structure

```
mlops-unit1/
│
├── data/               
├── src/                
│   ├── train_model.py
│   ├── mlflow_exp.py
│   ├── deployment.py
│   ├── monitor.py
│   ├── mnist_opt.py
│   └── train_cancer.py
├── models/            
├── .github/workflows/ 
├── requirements.txt   
└── README.md          
```

---

## 🧪 Experiments

### 🔹 Experiment 1: Git & Version Control

* Initialized Git repository
* Created Python script
* Committed changes
* Created and merged branch
* Pushed code to GitHub

---

### 🔹 Experiment 2: Machine Learning Workflow

* Loaded Iris dataset using sklearn
* Split data into training and testing sets
* Trained Logistic Regression model
* Evaluated model using accuracy and classification report
* Saved trained model

---

### 🔹 Experiment 3: Project Structure

* Organized files into MLOps-friendly structure
* Separated data, source code, and models
* Created requirements.txt and README.md

---

### 🔹 Experiment 4: MLflow Experiment Tracking

* Trained RandomForest model
* Logged parameters and metrics
* Saved model using MLflow
* Visualized results in MLflow UI

---

### 🔹 Experiment 5: DVC Versioning

* Initialized DVC
* Tracked dataset using `dvc add`
* Configured remote storage
* Pushed data using DVC

---

### 🔹 Experiment 6: CI/CD Pipeline

* Created GitHub Actions workflow
* Automated model training and evaluation
* Triggered pipeline on code push
* Ensured model performance using accuracy threshold

---

### 🔹 Experiment 8: Deployment Strategies

* Implemented A/B testing (Model A vs Model B)
* Compared models using accuracy
* Selected best model for deployment
* Created simple Flask API

---

### 🔹 Experiment 9: Monitoring & Logging

* Implemented logging using Python logging module
* Monitored model accuracy
* Created log file for tracking execution
* Implemented alert system for low accuracy

---

### 🔹 Experiment 10: Optimization & Security

* Trained MNIST model using TensorFlow
* Applied quantization (reduced precision)
* Simulated adversarial attack using noise
* Compared model performance before and after attack

---

## 🔁 Reproducibility Steps

1. Clone the repository:

   ```
   git clone https://github.com/Kumkum-Mishra/mlops-unit1.git
   cd mlops-unit1
   ```

2. Create virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

Example:

```
python src/train_model.py
python src/mlflow_exp.py
python src/deployment.py
python src/monitor.py
python src/mnist_opt.py
```

---

## 🧠 Technologies Used

* Python
* Pandas
* Scikit-learn
* MLflow
* DVC
* GitHub Actions (CI/CD)
* Flask
* TensorFlow

---

## 👨‍💻 Author

Kumkum Mishra
B.Tech CSE
2023340804

---

## 📌 Notes

* Demonstrates complete MLOps lifecycle
* Covers versioning, tracking, deployment, monitoring, and optimization
* Useful for understanding real-world ML pipelines

---
