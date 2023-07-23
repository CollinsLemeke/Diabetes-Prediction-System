Diabetes Prediction Using Logistic Regression
This repository contains a machine learning project focused on predicting the presence of diabetes based on various features using logistic regression. The dataset contains information about gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level, and the target variable "diabetes."

Dataset
The dataset (diabetes_data.csv) contains the following columns:

gender: The gender of the individual (e.g., Female, Male).
age: The age of the individual in years.
hypertension: Whether the individual has hypertension (0: No, 1: Yes).
heart_disease: Whether the individual has heart disease (0: No, 1: Yes).
smoking_history: The smoking history of the individual (e.g., never, current, No Info).
bmi: Body Mass Index of the individual.
HbA1c_level: The HbA1c level, a measure of blood glucose control, of the individual.
blood_glucose_level: The fasting blood glucose level of the individual.
diabetes: The target variable, indicating whether the individual has diabetes (0: No diabetes, 1: Diabetes).

Usage
You can use the trained logistic regression model to predict diabetes on new data. Prepare your data in the same format as the provided dataset, import the necessary libraries, and load the trained model. An example of how to use the model is provided in example_usage.py.

Model Training
The logistic regression model was trained on the dataset using Scikit-learn's LogisticRegression class. Hyperparameter tuning and cross-validation were performed to optimize the model's performance. The training code can be found in train_model.py.

Model Evaluation
The performance of the trained logistic regression model was evaluated using various metrics such as accuracy, precision, recall, and F1-score. The evaluation code can be found in evaluate_model.py.

Contributing
If you want to contribute to this project, feel free to fork the repository, create a new branch, make your changes, and submit a pull request.
