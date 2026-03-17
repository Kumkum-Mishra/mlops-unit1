# MLOps Unit 1 Lab

This repository contains all practical exercises for MLOps Unit 1, including Git version control, machine learning workflow, project organization, and reproducibility.

---

## 📁 Project Structure

```
mlops-unit1/
│
├── data/              
├── src/               
│   └── train_model.py
├── models/            
│   └── iris_model.pkl
├── requirements.txt   
└── README.md          
```

---

## 🧪 Experiments

### 🔹 Experiment 1: Git & Version Control

* Initialized Git repository
* Created Python script
* Committed changes
* Created and merged branch (experiment-v1)
* Pushed code to GitHub

---

### 🔹 Experiment 2: Machine Learning Workflow

* Loaded Iris dataset using sklearn
* Split data into training and testing sets
* Trained Logistic Regression model
* Evaluated model using accuracy and classification report
* Saved trained model using pickle

---

### 🔹 Experiment 3: Project Structure

* Organized files into standard MLOps structure
* Separated code, data, and models
* Created requirements.txt and README.md

---

### 🔹 Experiment 4: Reproducibility

* Environment setup instructions
* Clone and run project in new environment

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mlops-unit1.git
cd mlops-unit1
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python src/train_model.py
```

---

## 📊 Output

* Displays dataset preview
* Shows training and testing data size
* Prints model accuracy
* Displays classification report
* Saves trained model

---

## 💾 Model

* Model is saved in models/iris_model.pkl
* Can be reused for predictions

---

## 🧠 Technologies Used

* Python
* Pandas
* Scikit-learn
* Git & GitHub

---

## 👨‍💻 Author

Kumkum Mishra
B.Tech CSE
2023340804

---

## 📌 Notes

* This project demonstrates basic MLOps workflow
* Useful for understanding version control and ML pipeline
