# ML_Zoomcamp-Hmwk-6
JAMB Students Performance Prediction

Objective
Build and evaluate regression models to predict students' JAMB scores using the 2024 dataset from Kaggle.

Dataset
Download the dataset:

wget https://github.com/alexeygrigorev/datasets/raw/refs/heads/master/jamb_exam_results.csv

Data Preparation
Convert columns to lowercase:

df.columns = df.columns.str.lower().str.replace(' ', '_')

Remove student_id, fill missing values with zeros.

Split data (60%/20%/20%) using train_test_split(random_state=1).

Use DictVectorizer(sparse=True) for transformation.


Tasks Overview
Decision Tree Regressor: Train with max_depth=1 and identify the main feature for splitting.

Random Forest Regressor: Train with n_estimators=10 and evaluate RMSE on validation data.

Vary n_estimators: Check after which value (from 10 to 200) RMSE stops improving.

Best max_depth: Test [10, 15, 20, 25] with n_estimators up to 200 and find optimal max_depth.

Feature Importance: Identify the most important feature using feature_importances_.

XGBoost Tuning: Train with eta=0.3 and eta=0.1, compare RMSE results.
