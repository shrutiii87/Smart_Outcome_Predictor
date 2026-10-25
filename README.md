# 🎓 Smart Outcome Predictor

<!-- TODO: Replace with your banner image, e.g.
<img width="1200" height="500" src="https://github.com/user-attachments/assets/your-banner.png" />
-->

---

This project evaluates understanding of **supervised machine learning with ensemble methods** with a focus on how bagging, boosting, voting and stacking combine many models into one strong predictor. It covers the bias–variance trade-off, AdaBoost vs Gradient Boosting vs LightGBM vs XGBoost, and voting/stacking ensembles, and shows how to compare all of them on both a classification and a regression problem before choosing the best models for deployment.

---

## 📄 Problem Statement

You are working with an **online learning platform** that holds a dataset of **5,200 students** and wants a **Smart Outcome Predictor** that can predict, from student demographics, engagement and course attributes:

1. **Classification** — whether a student will **complete the course** (`completion_status`: 1 = completed, 0 = not completed)
2. **Regression** — the student's **final score** (`final_score`, 0–100)

Your task is to prepare the data, build bagging models, tune and compare AdaBoost, Gradient Boosting, LightGBM and XGBoost, evaluate voting and stacking ensembles, measure everything with accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE and R², and deliver a final deployment recommendation.

---

## ♻️ Workflow

<!-- TODO: Replace with your workflow image
<img width="1200" height="700" src="https://github.com/user-attachments/assets/your-workflow.png" />
-->

Data → Preprocessing → **Bagging** → **Boosting (AdaBoost / GB / LightGBM / XGBoost)** → **Voting & Stacking** → Evaluation → **Deploy the best model per task**

---

## 🌐 Live Demo

<!-- TODO: Add your Streamlit / web app link
[![Live App](https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge)](YOUR_APP_LINK)
-->

---

# 📂 Project Files

| 📄 File / Folder | 📌 Description |
|------------------|----------------|
| 📓 `Smart_Outcome_Predictor_Fixed_1.ipynb` | Main notebook — the complete, annotated bagging, boosting, voting, stacking and model-comparison pipeline |
| 📊 `Smart_Outcome_Predictor_Dataset_5200.csv.csv` | Raw student dataset (5,200 records) |
| 📘 `README.md` | Project documentation and workflow guide |
| 📂 `Visuals` | Folder of the output visuals/graphs |
| 📄 `Part_A_Conceptual_Foundation_Theory.pdf` | Project documentation of Part A (Theory) |
| 📋 `Part_G_Final_Analysis_Reporting_Report.pdf` | Report documentation of Part G (Final Report) |

---

## 🛠️ Tools Used

<div>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/LightGBM-2AAA5A?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Baseline-Bagging-059669?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Models-Boosting%20%7C%20Voting%20%7C%20Stacking-16A34A?style=for-the-badge"/>

</div>

---

## 🎬 Project Demo

<!-- TODO: Add your demo video link
[![Watch Demo](https://img.shields.io/badge/Watch%20Demo-click%20to%20view-blue?style=for-the-badge&logo=googledrive&logoColor=white)](YOUR_VIDEO_LINK)

📹 Click on the badge to watch the video.
-->

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
| `completion_status` | Integer | 1 = Completed, 0 = Not completed | 🎯 **Classification target** (imbalanced: ~62% / 38%) |
| `final_score` | Float | Final course score (35.2–100.0, mean ≈ 74.8) | 🎯 **Regression target** |

---

## 🧠 Part B : Dataset Understanding & Preparation

### 6️⃣ Identify features and target variables for both problems

```python
X = df.drop(columns=["student_id", "course_start_date", "completion_status", "final_score"])
y_classification = df["completion_status"]
y_regression     = df["final_score"]
```

**Output:** 15 input features · 2 targets — `completion_status` (classification) and `final_score` (regression)

💡 **Insight:** The dataset contains 15 input features, including age, country, device type, course level, sessions, quiz scores, and attendance rate. This dual-target setup allows the project to predict both course completion and final scores. 🎯

---

### 7️⃣ Perform a train-test split for classification and regression problems

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Output:** Training data (4,160, 15) · Testing data (1,040, 15) for both tasks

💡 **Insight:** Data is split 80/20 for both tasks with the same `random_state=42`, ensuring a consistent and reproducible split that allows fair evaluation of classification and regression models. 🔀

---

### 8️⃣ Apply basic preprocessing (scaling/encoding where required)

```python
median = X_train.median(numeric_only=True)
X_train = X_train.fillna(median);  X_test = X_test.fillna(median)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test  = pd.get_dummies(X_test,  drop_first=True).reindex(columns=X_train.columns, fill_value=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train);  X_test = scaler.transform(X_test)
```

**Output:** 273 missing values filled — `time_spent_hours` (112), `avg_quiz_score` (81), `attendance_rate` (80)

💡 **Insight:** Missing values are filled with the **training set median** to prevent leakage. One-hot encoding with `drop_first=True` plus column alignment keeps train/test consistent, and `StandardScaler` is fit only on training data for stable ensemble learning. 📐

---

## 📦 Part C : Bagging (Bootstrap Aggregating)

### 9️⃣ Implement Bagging Classifier for course completion prediction

```python
bagging_clf = BaggingClassifier(n_estimators=50, random_state=42)
bagging_clf.fit(X_train_class, y_train_class)
y_pred = bagging_clf.predict(X_test_class)
```

**Output:** Accuracy = **0.699**

💡 **Insight:** Bagging reduces variance by aggregating 50 bootstrap models, giving a stable prediction for course completion. 🎯

---

### 🔟 Implement Bagging Regressor for final score prediction

**Output:** MAE 8.05 · RMSE 10.08 · **R² 0.456**

💡 **Insight:** The model shows a moderate fit for `final_score`. RMSE > MAE indicates some larger errors, but bagging generalises better than a single regressor. 📊

---

### 1️⃣1️⃣ Compare bagging results with a single base model

| Model | Accuracy |
|-------|----------|
| Single Decision Tree | 0.645 |
| Bagging (50 trees) | **0.699** |

💡 **Insight:** Bagging improves performance by ~5 points by reducing overfitting and variance, proving the ensemble is more stable and accurate than a single base model. 📈

---

## ⚡️ Part D : Boosting Algorithms

### 1️⃣2️⃣–1️⃣3️⃣ AdaBoost Classifier & Regressor

**Output:** Accuracy **0.716** · MAE 8.56 · RMSE 10.52 · R² 0.407

💡 **Insight:** AdaBoost beats Bagging on classification by focusing sequentially on misclassified samples, but in regression it is sensitive to outliers and lands below Bagging. ⚡

---

### 1️⃣4️⃣ Analyze how weak learners improve sequentially

| # Learners | 1 | 25 | 50 |
|------------|-----|------|------|
| Accuracy | 0.655 | **0.721** | 0.716 |

💡 **Insight:** Accuracy improves as learners correct previous errors, but saturates around 25 learners — the optimal ensemble size before slight overfitting. 📈

---

### 1️⃣5️⃣–1️⃣6️⃣ Gradient Boosting Classifier & Regressor

**Output:** Accuracy **0.723** · MAE 7.85 · RMSE 9.81 · **R² 0.485**

💡 **Insight:** Gradient Boosting is the best model in both tasks — it builds trees sequentially on residual errors, giving the lowest errors and the highest R² for score prediction. 🏆

---

### 1️⃣7️⃣ Analyze learning rate and number of estimators

| Setting | Accuracy |
|---------|----------|
| learning rate 0.01 | 0.711 |
| **learning rate 0.1** | **0.723** |
| learning rate 0.2 | 0.712 |
| 50 estimators | 0.717 |
| **100 estimators** | **0.723** |
| 150 estimators | 0.719 |

💡 **Insight:** Learning rate 0.1 with 100 estimators is the optimal balance — a lower rate learns too slowly, a higher rate overshoots, and 150 estimators slightly overfits. ⚖️

---

### 1️⃣8️⃣–2️⃣0️⃣ LightGBM Classifier & Regressor + training efficiency

**Output:** Accuracy 0.711 (trained in **0.47 s**) · MAE 8.02 · RMSE 9.99 · R² 0.464

💡 **Insight:** LightGBM's leaf-wise growth and histogram splitting make it very fast, and it generalises almost as well as standard Gradient Boosting — a great speed/accuracy trade-off for large datasets. ⚙️

---

### 2️⃣1️⃣–2️⃣3️⃣ XGBoost Classifier & Regressor + robustness comparison

**Output:** Accuracy 0.680 · MAE 8.53 · RMSE 10.67 · R² 0.389

💡 **Insight:** With default settings XGBoost underperforms here (lowest of the four boosters in both tasks), showing that even strong algorithms need hyperparameter tuning per dataset. 📉

---

## 💼 Part E : Voting & Stacking Ensembles

### 2️⃣4️⃣–2️⃣5️⃣ Voting Classifier (Decision Tree + Random Forest + Logistic Regression) and Hard vs Soft Voting

| Voting type | Accuracy |
|-------------|----------|
| **Hard voting** | **0.732** |
| Soft voting | 0.678 |

💡 **Insight:** Hard voting combines diverse base learners by majority vote and is the **best classification result in the notebook**. Soft voting is dragged down by poorly calibrated base-model probabilities. 🗳️

---

### 2️⃣6️⃣–2️⃣7️⃣ Stacking Classifier & Regressor (meta-learner)

**Output:** Stacking Classifier accuracy **0.708** · Stacking Regressor MAE 8.09 · RMSE 10.12 · R² 0.451

💡 **Insight:** A meta-learner (Logistic Regression for classification, Linear Regression for regression) learns how much to trust each base model, but it did not beat simple majority voting for classification or single strong regressors for score prediction here. 🧩

---

## 📊 Part F : Model Evaluation & Comparison

### 2️⃣8️⃣ Evaluate classification models with Accuracy, Precision, Recall, F1 and ROC-AUC

| Metric | Value (Random Forest) |
|--------|-----------------------|
| Accuracy | 0.714 |
| Precision | 0.654 |
| Recall | 0.516 |
| F1-Score | 0.577 |
| ROC-AUC | 0.761 |

💡 **Insight:** High precision with moderate recall shows the model predicts the positive class correctly but misses some completions; ROC-AUC 0.76 indicates good overall discriminative ability. 🎯

---

### 2️⃣9️⃣ Evaluate regression models with MAE, RMSE and R² Score

**Output:** Random Forest Regressor — MAE 7.99 · RMSE 10.00 · R² 0.463

💡 **Insight:** Low MAE and RMSE with a high R² shows strong generalisation and the ability to capture non-linear patterns in student data. 📊

---

### 3️⃣0️⃣ Compare all ensemble techniques and identify the best model for each task

| Model | Accuracy (Classification) | R² (Regression) |
|-------|---------------------------|-----------------|
| **Voting (hard)** | **0.732** 🏆 | — |
| Gradient Boosting | 0.723 | **0.485** 🏆 |
| AdaBoost | 0.716 | 0.407 |
| Random Forest | 0.714 | 0.463 |
| LightGBM | 0.711 | 0.464 |
| Stacking | 0.708 | 0.451 |
| Bagging | 0.699 | 0.456 |
| XGBoost | 0.680 | 0.389 |

💡 **Insight:** Different ensemble strategies are optimal for each task — the **Voting Classifier** wins classification by combining diverse classifiers through majority vote, while **Gradient Boosting** wins regression by sequentially reducing residual errors. 🥇

---

## 📂 Project Workflow

1. **Theory (Part A)** → Ensemble learning, bagging vs boosting vs voting, bias–variance trade-off, stacking, AdaBoost / Gradient Boosting / LightGBM / XGBoost
2. **Dataset Understanding** → Overview, missing values, features and dual targets
3. **Preprocessing** → Median imputation + one-hot encoding + leakage-free scaling
4. **Train/Test Split** → 80/20 split (4,160 / 1,040 students) for both tasks
5. **Bagging** → Bagging Classifier & Regressor, comparison with a single decision tree
6. **Boosting** → AdaBoost, Gradient Boosting, LightGBM and XGBoost with tuning analysis
7. **Voting & Stacking** → Hard vs soft voting, stacking with meta-learners
8. **Model Evaluation** → Accuracy, Precision, Recall, F1, ROC-AUC, MAE, RMSE and R²
9. **Final Report** → Bagging vs boosting impact, boosting comparison, deployment recommendation

---

## 📈 Results & Insights

- ✅ **Eight models** built and compared — Bagging, AdaBoost, Gradient Boosting, LightGBM, XGBoost, Random Forest, Voting and Stacking
- ✅ **Voting Classifier (hard)** is the best classification model — accuracy **73.2%**
- ✅ **Gradient Boosting** is the best regression model — R² **0.485**, MAE 7.85, RMSE 9.81
- ✅ **Bagging beats a single tree** (0.699 vs 0.645) by reducing variance; boosting mainly reduces bias
- ✅ **Not every booster beats bagging** — AdaBoost, Gradient Boosting and LightGBM do; XGBoost (untuned) does not
- ✅ **LightGBM trains in under half a second** with near-Gradient-Boosting accuracy
- ✅ **Recommendation:** deploy the hard Voting Classifier for completion prediction and the Gradient Boosting Regressor for score prediction; support teachers' decisions rather than replace them

---

## 📌 Expected Outcomes

- Understand **ensemble learning** and how bagging, boosting, voting and stacking differ
- Apply the **bias–variance trade-off** to explain why ensembles outperform single models
- Build and tune **bagging ensembles and four boosting algorithms** for both classification and regression
- Evaluate models with **accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE and R²**
- Compare **variance-reducing, bias-reducing and meta-learning** strategies and explain the trade-offs
- Select a final model per task and write a **deployment recommendation**

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

## 🚀 Future Scope

- [ ] Tune XGBoost's `learning_rate`, `max_depth` and `n_estimators` to close the gap with Gradient Boosting
- [ ] Confirm the results with cross-validation instead of a single train-test split
- [ ] Calibrate probabilities before soft voting, then re-test hard vs soft voting
- [ ] Engineer engagement features (e.g. sessions per week, quiz score trends)
- [ ] Handle the 62/38 class imbalance with class weights or resampling
- [ ] Tune the classification threshold to balance missed completions against false alarms
- [ ] Deploy the models behind a simple dashboard for teachers with prediction explanations

---

## 🙏 Thank You

Thank you for taking the time to explore this project!
Your feedback, suggestions, and contributions are always welcome.

⭐ If you found this project helpful, don't forget to **star the repository** and share it with others.
