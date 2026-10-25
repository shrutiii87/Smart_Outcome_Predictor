# 🎓 Smart Outcome Predictor

<img width="1200" alt="Smart Outcome Predictor" src="https://github.com/user-attachments/assets/22a08776-0809-490d-8ba3-2aea6c4909ef" />

A supervised machine learning project that uses **Bagging, Boosting, Voting, and Stacking** to predict student course completion and final scores.

## 📄 Problem Statement

The dataset contains 5,200 student records. The project solves two prediction tasks:

- **Classification:** Predict whether a student completes a course (`completion_status`).
- **Regression:** Predict the student's final score (`final_score`).

## 🎬 Project Demo

[![Watch Video Demo](https://img.shields.io/badge/Watch_Video_Demo-70499A?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1oD4nGN8-yN_oPSTUD4Knz48nX6V0lk5o/view?usp=sharing)

[![Live Demo](https://img.shields.io/badge/Live_Demo-70499A?style=for-the-badge&logo=streamlit&logoColor=white)](YOUR_STREAMLIT_APP_URL)

## 🛠️ Tools Used

<img src="https://img.shields.io/badge/Python-70499A?style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/Jupyter_Notebook-white?style=for-the-badge&logo=jupyter&logoColor=70499A" /> <img src="https://img.shields.io/badge/Pandas-70499A?style=for-the-badge&logo=pandas&logoColor=white" /> <img src="https://img.shields.io/badge/NumPy-white?style=for-the-badge&logo=numpy&logoColor=70499A" /> <img src="https://img.shields.io/badge/Scikit--learn-70499A?style=for-the-badge&logo=scikitlearn&logoColor=white" /> <img src="https://img.shields.io/badge/Matplotlib-white?style=for-the-badge&logo=matplotlib&logoColor=70499A" /> <img src="https://img.shields.io/badge/XGBoost-70499A?style=for-the-badge&logo=xgboost&logoColor=white" /> <img src="https://img.shields.io/badge/LightGBM-white?style=for-the-badge&logo=lightgbm&logoColor=70499A" />

## 📂 Project Files

| File / Folder | Description |
|---|---|
| `Smart_Outcome_Predictor_Fixed_1.ipynb` | Main Jupyter Notebook |
| `Smart_Outcome_Predictor_Dataset_5200.csv.csv` | Student dataset |
| `Visuals/` | Graphs and visualizations |
| `Part_A_Conceptual_Foundation_Theory.pdf` | Theory documentation |
| `Part_G_Final_Analysis_Reporting_Report.pdf` | Final project report |

## 📊 Dataset Overview

- **Total records:** 5,200
- **Total columns:** 19
- **Input features:** 15
- **Classification target:** `completion_status`
- **Regression target:** `final_score`

Important features include age, country, device type, course category, sessions, time spent, videos watched, quiz scores, assignments, forum posts, and attendance rate.

## 🧪 Part B: Data Understanding & Preparation

### 1. Select Features and Targets

```python
X = df.drop(columns=[
    "student_id", "course_start_date",
    "completion_status", "final_score"
], errors="ignore")

y_classification = df["completion_status"]
y_regression = df["final_score"]
```

### 2. Train-Test Split

```python
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
```

The split uses 80% training data (4,160 records) and 20% testing data (1,040 records).

### 3. Preprocessing

Missing numerical values are filled using training-set medians. Categorical features are one-hot encoded, and numerical features are scaled using `StandardScaler`.

```python
median = X_train.median(numeric_only=True)
X_train = X_train.fillna(median)
X_test = X_test.fillna(median)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Missing values were found in `time_spent_hours` (112), `avg_quiz_score` (81), and `attendance_rate` (80).

## 📦 Part C: Bagging

Bagging trains multiple models on bootstrap samples and combines their predictions.

### Bagging Classifier

```python
model = BaggingClassifier(n_estimators=50, random_state=42)
model.fit(X_train_class, y_train_class)
pred = model.predict(X_test_class)

print("Accuracy:", accuracy_score(y_test_class, pred))
```

### Bagging Regressor

```python
model = BaggingRegressor(n_estimators=50, random_state=42)
model.fit(X_train_reg, y_train_reg)
pred = model.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred)))
print("R2:", r2_score(y_test_reg, pred))
```

## ⚡ Part D: Boosting Algorithms

The project implements AdaBoost, Gradient Boosting, LightGBM, and XGBoost for classification and regression.

```python
model = GradientBoostingClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train_class, y_train_class)
pred = model.predict(X_test_class)

print("Accuracy:", accuracy_score(y_test_class, pred))
```

Boosting builds models sequentially, allowing later learners to improve on earlier errors.

## 🤝 Part E: Voting & Stacking

### Voting Classifier

```python
model = VotingClassifier(
    estimators=[
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("lr", LogisticRegression(max_iter=1000))
    ],
    voting="hard"
)

model.fit(X_train_class, y_train_class)
pred = model.predict(X_test_class)

print("Accuracy:", accuracy_score(y_test_class, pred))
```

### Stacking Classifier

```python
model = StackingClassifier(
    estimators=[
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=50, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000)
)

model.fit(X_train_class, y_train_class)
pred = model.predict(X_test_class)

print("Accuracy:", accuracy_score(y_test_class, pred))
```

Voting combines predictions from multiple models, while stacking uses a meta-learner to learn from base-model predictions.

## 📈 Part F: Model Evaluation & Comparison

Classification is evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC. Regression is evaluated using MAE, RMSE, and R².

### Classification Results

| Model | Accuracy |
|---|---:|
| Bagging | 69.81% |
| AdaBoost | 71.63% |
| Gradient Boosting | 72.31% |
| LightGBM | 71.15% |
| XGBoost | 68.08% |
| Voting | **73.17%** |
| Stacking | 70.87% |
| Random Forest | 71.44% |

### Regression Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Bagging | 8.05 | 10.08 | 0.4558 |
| AdaBoost | 8.57 | 10.52 | 0.4072 |
| Gradient Boosting | **7.85** | **9.81** | **0.4854** |
| LightGBM | 8.03 | 10.00 | 0.4649 |
| XGBoost | 8.54 | 10.68 | 0.3898 |
| Stacking | 8.09 | 10.12 | 0.4517 |
| Random Forest | 8.00 | 10.01 | 0.4638 |

### Final Findings

- **Best classification model:** Voting Classifier, with approximately 73.17% accuracy.
- **Best regression model:** Gradient Boosting Regressor, with R² of 0.4854.
- Different ensemble methods perform differently depending on the prediction task.

## 🔄 Project Workflow

1. Theory and ensemble learning concepts
2. Dataset understanding and target selection
3. Missing-value handling, encoding, and scaling
4. Train-test split
5. Bagging implementation
6. Boosting algorithms and comparisons
7. Voting and stacking
8. Model evaluation and final analysis

## ⚙️ Installation & Setup

```bash
git clone https://github.com/shrutiii87/Smart_Outcome_Predictor.git
cd Smart_Outcome_Predictor

pip install pandas numpy scikit-learn xgboost lightgbm matplotlib jupyter

jupyter notebook Smart_Outcome_Predictor_Fixed_1.ipynb
```

## 🙏 Thank You

Thank you for exploring **Smart Outcome Predictor**!

⭐ If you find this project useful, consider starring the repository.
