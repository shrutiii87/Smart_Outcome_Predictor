# 🎓 Smart Outcome Predictor

<img width="1200" height="520" alt="679098054-2f0a9f1e-daaa-4daa-b4fd-aba6f942eac1" src="https://github.com/user-attachments/assets/22a08776-0809-490d-8ba3-2aea6c4909ef" />


---

This project is built on ensemble learning in supervised machine learning. It covers bagging, boosting, voting and stacking, compares how each family of models combines many learners into one strong predictor, and applies all of them to a real dataset of online learning platform students to predict both course completion and final scores.

---

## 📄 Problem Statement

<table>
<tr>
<td width="50%" valign="top">

You are working with an online learning platform that holds a dataset of **5,200 students**. The platform wants a **Smart Outcome Predictor** that can predict, from student demographics, engagement and course attributes:

**1. Classification** — whether a student will complete the course (completion_status: 1 = completed, 0 = not completed)

**2. Regression** — the student's final score (final_score, 35.2–100.0)

The task is to prepare the data, build bagging models, implement and analyse **AdaBoost, Gradient Boosting, LightGBM and XGBoost**, evaluate **voting and stacking ensembles**, measure everything with **accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE and R2**, and identify the best performing model for each task.

</td>
<td width="50%" valign="top">

<img src="https://github.com/user-attachments/assets/ecca4c77-e990-43b1-8452-ec82ae083a1a" width="100%">

</td>
</tr>
</table>


---

## 📂 Project Files

| 📄 File / Folder | 📌 Description |
|------------------|----------------|
| 📓 `Smart_Outcome_Predictor_Fixed_1.ipynb` | Main notebook — the complete, annotated bagging, boosting, voting, stacking and model-comparison pipeline |
| 📊 `Smart_Outcome_Predictor_Dataset_5200.csv.csv` | Raw student dataset (5,200 records) |
| 📘 `README.md` | Project documentation and workflow guide |
| 📂 `Visuals` | Folder of the output visuals/graphs |
| 📄 `Part_A_Conceptual_Foundation_Theory.pdf` | Project documentation of Part A (Theory) |
| 📋 `Part_G_Final_Analysis_Reporting_Report.pdf` | Report documentation of Part G (Final Report) |

---

## 🛠 Tools Used

<div>

<img src="https://img.shields.io/badge/Python-8B5CF6?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter%20Notebook-5B21B6?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-A78BFA?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-6D28D9?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-7C3AED?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-7B2CBF?style=for-the-badge&logo=matplotlib&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-4B1D7F?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/LightGBM-9333EA?style=for-the-badge&logo=lightgbm&logoColor=white"/>
<img src="https://img.shields.io/badge/Bagging-6A0DAD?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Boosting-9575CD?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Voting-8E44AD?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Stacking-7E57C2?style=for-the-badge&logoColor=white"/>

</div>

---

## 🎬 Project Demo

<a href="https://drive.google.com/file/d/1oD4nGN8-yN_oPSTUD4Knz48nX6V0lk5o/view?usp=sharing">
  <img src="https://img.shields.io/badge/WATCH%20DEMO-ADD%20YOUR%20LINK-C4B5FD?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=4B1D7F"/>
</a>

---

## 🌐 Live Demo

<a href="https://messageintelligencesystem-fny7qy4goeazapplv84pxzk.streamlit.app/">
  <img src="https://img.shields.io/badge/LIVE%20DEMO-OPEN%20APP-D8B4FE?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=5B21B6"/>
</a>

---

### 🧬 Dataset Structure — Smart Outcome Predictor Dataset

| Field Name | Data Type | Description | Notes |
|------------|-----------|-------------|-------|
| `student_id` | Integer | Unique identifier for each student | Identifier only — not a real feature |
| `age` | Integer | Age of the student | Model feature |
| `country_region` | Object | Region of the student | Model feature (one-hot encoded) |
| `device_type` | Object | Device used for learning | Model feature (one-hot encoded) |
| `education_background` | Object | Educational background | Model feature (one-hot encoded) |
| `course_level` | Object | Course difficulty level | Model feature (one-hot encoded) |
| `course_category` | Object | Course subject category | Model feature (one-hot encoded) |
| `course_start_date` | Object (date) | Date the course started | Not used as a model feature |
| `week_of_year` | Integer | Week of the year the course started | Model feature |
| `sessions` | Integer | Number of learning sessions | Model feature |
| `time_spent_hours` | Float | Total hours spent on the course | Model feature — has missing values |
| `videos_watched` | Integer | Number of videos watched | Model feature |
| `quiz_attempts` | Integer | Number of quiz attempts | Model feature |
| `assignments_submitted` | Integer | Number of assignments submitted | Model feature |
| `forum_posts` | Integer | Number of forum posts | Model feature |
| `avg_quiz_score` | Float | Average quiz score | Model feature — has missing values |
| `attendance_rate` | Float | Attendance rate | Model feature — has missing values |
| `completion_status` | Integer | 1 = Completed, 0 = Not completed | 🎯 Classification target (62.5% not completed / 37.5% completed) |
| `final_score` | Float | Final course score (35.2–100.0, mean ≈ 74.8) | 🎯 Regression target |

---

<img width="2048" height="1152" alt="861255692_1455238460996258_2734771313755867768_n" src="https://github.com/user-attachments/assets/c6afa8d6-7b30-4ec2-b094-4feec7574298" />


---

## 🧪 Part B: Dataset Understanding & Preparation

<img width="1200" height="470" alt="679111918-b7450104-e36c-4117-9e1f-2acc3df738ee" src="https://github.com/user-attachments/assets/59f5e2a2-e12b-478b-9171-d172d8856fce" />


### 6. Identify features and target variables for both problems.

```python
X = df.drop(columns=["student_id", "course_start_date",
                     "completion_status", "final_score"], errors="ignore")
y_classification = df["completion_status"]
y_regression     = df["final_score"]
```

🎯 Insight: The dataset contains 15 input features, including age, country, device type, course level, sessions, quiz scores, and attendance rate. The classification target is completion_status, while the regression target is final_score. This setup allows the project to predict both course completion and final scores.

---

### 7. Perform a train-test split for classification and regression problems.

```python
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X, y_classification, test_size=0.2, random_state=42)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42)
```

🎯 Insight: Data is split into 80% training (4160 samples) and 20% testing (1040 samples) for both tasks with 15 features. Same random_state=42 ensures consistent and reproducible split, allowing fair evaluation of classification and regression models.

---

### 8. Apply basic preprocessing (scaling/encoding if required).

```python
def prepare(X_train, X_test):
    median = X_train.median(numeric_only=True)
    X_train = X_train.fillna(median)
    X_test  = X_test.fillna(median)

    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test  = pd.get_dummies(X_test,  drop_first=True)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test
```

🎯 Insight: Total 273 missing values detected - time_spent_hours (112), avg_quiz_score (81), attendance_rate (80) - filled with training set median to prevent leakage. Then one-hot encoding with drop_first=True and column alignment ensures train/test consistency, and StandardScaler is fit only on train data to scale features for stable ensemble learning.

---

## 📦 Part C: Bagging (Bootstrap Aggregating)

<img width="1200" height="450" alt="679116947-d6654a7a-4777-4b1c-8630-c0d87d1e231f" src="https://github.com/user-attachments/assets/375873c3-e03f-4461-a534-20715aa4a886" />


### 9. Implement Bagging Classifier for course completion prediction.

```python
model = BaggingClassifier(n_estimators=50, random_state=42)
model.fit(X_train_class, y_train_class)
pred = model.predict(X_test_class)
bagging_acc = accuracy_score(y_test_class, pred)
```

🎯 Insight: Accuracy is 0.6987 with n_estimators=50. Bagging reduces variance by aggregating 50 bootstrap models, giving stable prediction for course completion.

---

### 10. Implement Bagging Regressor for final score prediction.

```python
model = BaggingRegressor(n_estimators=50, random_state=42)
model.fit(X_train_reg, y_train_reg)
pred = model.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred)))
print("R2 Score:", r2_score(y_test_reg, pred))
```

🎯 Insight: MAE is 8.05, RMSE is 10.08 and R2 is 0.4558. Model shows moderate fit for final_score prediction, RMSE > MAE indicates some larger errors, but bagging helps to generalize better than single regressor.

---

### 11. Compare bagging results with a single base model.

```python
single = DecisionTreeClassifier(random_state=42)
single.fit(X_train_class, y_train_class)

bagging = BaggingClassifier(n_estimators=50, random_state=42)
bagging.fit(X_train_class, y_train_class)

print("Single Model Accuracy:", accuracy_score(y_test_class, single.predict(X_test_class)))
print("Bagging Accuracy:",     accuracy_score(y_test_class, bagging.predict(X_test_class)))
```

🎯 Insight: Single Decision Tree gives 0.645 accuracy while Bagging gives 0.698 accuracy. Bagging improves performance by ~5% by reducing overfitting and variance, proving ensemble is more stable and accurate than a single base model.

---

## ⚡️ Part D: Boosting Algorithms

<img width="1200" height="470" alt="679143826-6b7f11cc-efc2-400f-b98e-9d59e1d95595" src="https://github.com/user-attachments/assets/b745b6c9-d89e-4412-92d1-998a208f7533" />


## AdaBoost

### 12. Implement AdaBoost Classifier.

```python
model = AdaBoostClassifier(random_state=42)
model.fit(X_train_class, y_train_class)
pred = model.predict(X_test_class)
adaboost_acc = accuracy_score(y_test_class, pred)
```

🎯 Insight: AdaBoost achieves 0.716 accuracy, higher than Bagging. It improves by focusing sequentially on misclassified samples, giving more weight to hard cases, which boosts overall prediction performance.

---

### 13. Implement AdaBoost Regressor.

```python
model = AdaBoostRegressor(random_state=42)
model.fit(X_train_reg, y_train_reg)
pred = model.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred)))
print("R2 Score:", r2_score(y_test_reg, pred))
```

🎯 Insight: AdaBoost Regressor gives MAE 8.56, RMSE 10.52 and R2 0.407. R2 is lower than Bagging Regressor, showing AdaBoost focuses more on classification here and is sensitive to outliers in regression, leading to slightly higher error.

---

### 14. Analyze how weak learners improve sequentially.

```python
model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X_train_class, y_train_class)

for i, pred in enumerate(model.staged_predict(X_test_class), start=1):
    if i in [1, 10, 25, 50]:
        print("Learners:", i, "Accuracy:", accuracy_score(y_test_class, pred))
```

🎯 Insight: Accuracy improves from 0.655 with 1 learner to 0.721 with 25 learners, then slightly drops to 0.716 at 50 learners. This shows weak learners improve sequentially by correcting previous errors, but after 25 learners performance saturates, indicating optimal ensemble size.

---

## Gradient Boosting

### 15. Implement Gradient Boosting Classifier.

```python
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train_class, y_train_class)
pred = model.predict(X_test_class)
gb_acc = accuracy_score(y_test_class, pred)
```

🎯 Insight: Gradient Boosting Classifier gives accuracy 0.723, which is higher than Bagging and AdaBoost. It improves by building trees sequentially on residual errors, giving better generalization and highest accuracy so far for classification.

---

### 16. Implement Gradient Boosting Regressor.

```python
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train_reg, y_train_reg)
pred = model.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred)))
print("R2 Score:", r2_score(y_test_reg, pred))
```

🎯 Insight: Gradient Boosting Regressor gives MAE 7.85, RMSE 9.80 and R2 0.485, which is better than both Bagging and AdaBoost Regressor. It reduces errors by sequentially learning from residuals, giving lowest error and highest R2 so far for final score prediction.

---

### 17. Analyze learning rate and number of estimators.

```python
for rate in [0.01, 0.1, 0.2]:
    model = GradientBoostingClassifier(
        learning_rate=rate, n_estimators=100, random_state=42)
    model.fit(X_train_class, y_train_class)
    print("Learning Rate:", rate, "Accuracy:", accuracy_score(y_test_class, model.predict(X_test_class)))

for n in [50, 100, 150]:
    model = GradientBoostingClassifier(
        learning_rate=0.1, n_estimators=n, random_state=42)
    model.fit(X_train_class, y_train_class)
    print("Estimators:", n, "Accuracy:", accuracy_score(y_test_class, model.predict(X_test_class)))
```

🎯 Insight: Best learning rate is 0.1 with accuracy 0.723, lower rate 0.01 learns too slowly and higher rate 0.2 causes overshooting. For estimators, 100 gives highest accuracy 0.723, 50 is underfitting and 150 shows slight overfitting, so learning rate 0.1 and 100 estimators is the optimal balance.

---

## Light Gradient Boosting

### 18. Implement LightGBM Classifier.

🎯 Insight: LightGBM Classifier gives accuracy 0.711, slightly lower than Gradient Boosting. It uses leaf-wise growth which is faster and efficient, but in this dataset it generalizes a bit less than standard gradient boosting.

---

### 19. Implement LightGBM Regressor.

🎯 Insight: LightGBM Regressor gives MAE 8.02, RMSE 9.99 and R2 0.464, which is better than AdaBoost but slightly lower than Gradient Boosting. It shows fast and balanced performance with moderate generalization for final score prediction.

---

### 20. Analyze performance and training efficiency.

🎯 Insight: LightGBM Classifier achieves accuracy 0.711 with training time of only 0.47 seconds, showing high training efficiency. It gives good accuracy with very fast training due to leaf-wise growth and histogram based splitting, making it suitable for large datasets.

---

## XGBoost

### 21. Implement XGBoost Classifier.

🎯 Insight: XGBoost Classifier gives accuracy 0.680, which is lower than Gradient Boosting and LightGBM for this dataset. Despite its regularized boosting and high performance on large datasets, it shows slight underperformance here, indicating need for hyperparameter tuning.

---

### 22. Implement XGBoost Regressor.

🎯 Insight: XGBoost Regressor gives MAE 8.53, RMSE 10.67 and R2 0.389, which is lower than Gradient Boosting and LightGBM Regressor. It shows moderate performance for final score prediction with higher error, indicating it needs tuning of learning rate and depth for this regression task.

---

### 23. Compare performance and robustness with other boosting models.

```python
models = {
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbosity=-1),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
}

for name, model in models.items():
    model.fit(X_train_class, y_train_class)
    print(name, "Accuracy:", accuracy_score(y_test_class, model.predict(X_test_class)))
```

🎯 Insight: Comparison shows Gradient Boosting gives highest accuracy 0.723, followed by AdaBoost 0.716, LightGBM 0.711 and XGBoost 0.680. This shows Gradient Boosting is most robust for classification here, while XGBoost underperforms without tuning, and LightGBM gives balanced speed and accuracy.

---

## 💼 Part E: Voting & Stacking Ensembles

<img width="1200" height="470" alt="683371061-0797ce15-e251-4569-8938-e7c3a98cd308" src="https://github.com/user-attachments/assets/05427d5c-ec4b-47c3-91ab-ba8b94d2da5d" />


### 24. Implement Voting Classifier using multiple base classifiers.

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
voting_acc = accuracy_score(y_test_class, pred)
```

🎯 Insight: Voting Classifier with DT, Random Forest and Logistic Regression gives accuracy 0.731, higher than individual boosting models. Hard voting combines diverse base learners and reduces individual model bias, giving improved and more robust classification performance.

---

### 25. Compare Hard Voting vs Soft Voting.

```python
hard = VotingClassifier(estimators=[...], voting="hard")
soft = VotingClassifier(estimators=[...], voting="soft")

hard.fit(X_train_class, y_train_class)
soft.fit(X_train_class, y_train_class)

print("Hard Voting Accuracy:", accuracy_score(y_test_class, hard.predict(X_test_class)))
print("Soft Voting Accuracy:", accuracy_score(y_test_class, soft.predict(X_test_class)))
```

🎯 Insight: Hard Voting gives accuracy 0.731 while Soft Voting gives 0.678, showing hard voting performs better here. Hard voting uses majority class labels which works well when base models are confident, while soft voting averages probabilities and can be affected by poorly calibrated probabilities of base models in this dataset.

---

### 26. Implement Stacking Classifier with a meta-learner.

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
stacking_acc = accuracy_score(y_test_class, pred)
```

🎯 Insight: Stacking Classifier with DT and Random Forest as base learners and Logistic Regression as meta-learner gives accuracy 0.708. It is slightly lower than Voting hard 0.731, showing that meta-learner could not combine predictions as effectively as simple majority voting for this dataset, but still provides robust generalization by learning from base model outputs.

---

### 27. Implement Stacking Regressor for score prediction.

```python
model = StackingRegressor(
    estimators=[
        ("dt", DecisionTreeRegressor(random_state=42)),
        ("rf", RandomForestRegressor(n_estimators=50, random_state=42))
    ],
    final_estimator=LinearRegression()
)
model.fit(X_train_reg, y_train_reg)
pred = model.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred)))
print("R2 Score:", r2_score(y_test_reg, pred))
```

🎯 Insight: Stacking Regressor with DT and Random Forest as base learners and Linear Regression as meta-learner gives MAE 8.09, RMSE 10.12 and R2 0.451. Performance is similar to Gradient Boosting Regressor, showing stacking improves generalization for regression by combining base predictions through meta-learner, but does not significantly outperform single strong regressors for score prediction.

---

## 📊 Part F: Model Evaluation & Comparison

<img width="1200" height="470" alt="683372101-91c7faca-2dad-49e7-bbff-9fe80d9ea4c9" src="https://github.com/user-attachments/assets/a532a3b2-199e-4a15-afe6-0adc2a045ba0" />


### 28. Evaluate classification models using Accuracy, Precision, Recall, F1-Score and ROC-AUC.

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_class, y_train_class)
pred = model.predict(X_test_class)
prob = model.predict_proba(X_test_class)[:, 1]

print("Accuracy:",  accuracy_score(y_test_class, pred))
print("Precision:", precision_score(y_test_class, pred, zero_division=0))
print("Recall:",    recall_score(y_test_class, pred, zero_division=0))
print("F1-Score:",  f1_score(y_test_class, pred, zero_division=0))
print("ROC-AUC:",   roc_auc_score(y_test_class, prob))
```

🎯 Insight: Random Forest Classifier with full evaluation gives Accuracy 0.714, Precision 0.654, Recall 0.516, F1-Score 0.577 and ROC-AUC 0.761. High precision with moderate recall shows model is good at predicting positive class correctly but misses some positives. ROC-AUC 0.76 indicates good overall discriminative ability, and F1 score balances precision and recall for final assessment.

---

### 29. Evaluate regression models using MAE, RMSE and R2 Score.

```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_reg, y_train_reg)
pred = model.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, pred)))
print("R2 Score:", r2_score(y_test_reg, pred))
```

🎯 Insight: Random Forest Regressor gives MAE 7.99, RMSE 10.00 and R2 0.463, which is one of the best among regressors for score prediction. Lower MAE and RMSE with higher R2 shows strong generalization and ability to capture non-linear patterns in student data, making it more reliable than linear models for final score prediction.

---

### 30. Compare all ensemble techniques and identify the best performing model for each task.

```python
class_results = {
    "Bagging": bagging_acc, "AdaBoost": adaboost_acc,
    "Gradient Boosting": gb_acc, "LightGBM": lgbm_acc,
    "XGBoost": xgb_acc, "Voting": voting_acc,
    "Stacking": stacking_acc, "Random Forest": rf_acc
}
best_class = max(class_results, key=class_results.get)

reg_results = {
    "Bagging": bagging_r2, "AdaBoost": adaboost_r2,
    "Gradient Boosting": gb_r2, "LightGBM": lgbm_r2,
    "XGBoost": xgb_r2, "Stacking": stacking_r2,
    "Random Forest": rf_r2
}
best_reg = max(reg_results, key=reg_results.get)
```

🎯 Insight: Final comparison shows Voting Classifier as best classification model with 0.7317 accuracy and Gradient Boosting as best regression model with R2 0.4854. Voting excels by combining diverse classifiers through majority vote, while Gradient Boosting gives best regression performance by sequentially reducing residual errors. This shows different ensemble strategies are optimal for classification vs regression on this dataset.

---

### 📊 All Grpahs :- 

<img width="960" height="540" alt="Screenshot 2026-10-26 074256" src="https://github.com/user-attachments/assets/e0d1a730-d0b7-4b9d-904a-448b99c47694" />


---

## 📂 Project Workflow

1. Theory (Part A) → Ensemble learning, bagging vs boosting vs voting, bias–variance trade-off, stacking, AdaBoost / Gradient Boosting / LightGBM / XGBoost
2. Dataset Understanding → Overview, missing values, features and dual targets
3. Preprocessing → Median imputation + one-hot encoding + leakage-free scaling
4. Train/Test Split → 80/20 split (4,160 / 1,040 students) for both tasks
5. Bagging → Bagging Classifier & Regressor, comparison with a single decision tree
6. Boosting → AdaBoost, Gradient Boosting, LightGBM and XGBoost with tuning analysis
7. Voting & Stacking → Hard vs soft voting, stacking with meta-learners
8. Model Evaluation → Accuracy, Precision, Recall, F1, ROC-AUC, MAE, RMSE and R2
9. Final Report → Bagging vs boosting impact, boosting comparison, deployment recommendation

---

## ⚙️ Installation & Setup

```bash
# clone the repository
git clone https://github.com/yourusername/smart-outcome-predictor.git
cd smart-outcome-predictor

# create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib jupyter

# launch the notebook
jupyter notebook Smart_Outcome_Predictor_Fixed_1.ipynb
```

---

## 🙏 Thank You

Thank you for taking the time to explore this project!
Your feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don't forget to star the repository and share it with others.
