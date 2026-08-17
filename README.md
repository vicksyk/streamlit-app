# streamlit-app
# Customer Churn Prediction Using Machine Learning

## Student Details

- **Name:** Vikas Kamble
- **BITS ID:** 2025AC05906
- **Course:** Machine Learning
- **Assignment:** Machine Learning Assignment 2

## 1. Problem Statement

Customer churn is a major concern for telecommunication companies because retaining existing customers is generally important for maintaining business continuity and revenue.

The objective of this project is to develop and compare multiple machine learning classification models for predicting whether a telecom customer is likely to churn. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The project also provides an interactive Streamlit web application through which users can upload test data, select a classification model, view evaluation metrics, and examine a confusion matrix.

## 2. Dataset Description

The project uses the Telco Customer Churn dataset.

- **Dataset type:** Binary classification
- **Number of instances:** 7,043
- **Target variable:** Churn
- **Target classes:** No Churn and Churn
- **Dataset file:** `Telco-Customer-Churn.csv`

The dataset contains customer demographic information, account details, subscribed services, contract type, payment method, monthly charges, total charges, tenure, and churn status.

The target variable indicates whether a customer discontinued the telecom service.

## 3. Repository and Application Links

- **GitHub Repository:** https://github.com/vicksyk/streamlit-app
- **Live Streamlit Application:** https://app-app-hzck8ncpkv3ncw8bwqgisp.streamlit.app/

## 4. Models Implemented

The following six classification models were implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest Classifier
6. Support Vector Machine

## 5. Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8070 | 0.8416 | 0.6584 | 0.5668 | 0.6092 | 0.4843 |
| Decision Tree | 0.7417 | 0.6623 | 0.5139 | 0.4947 | 0.5041 | 0.3296 |
| K-Nearest Neighbors | 0.7473 | 0.7718 | 0.5253 | 0.5000 | 0.5123 | 0.3422 |
| Naive Bayes | 0.6558 | 0.8096 | 0.4269 | 0.8663 | 0.5719 | 0.3951 |
| Random Forest | 0.7899 | 0.8265 | 0.6336 | 0.4947 | 0.5556 | 0.4263 |
| Support Vector Machine | 0.7928 | 0.7961 | 0.6444 | 0.4893 | 0.5562 | 0.4312 |

## 6. Model Performance Observations

| ML Model | Observation |
|---|---|
| Logistic Regression | Achieved the highest Accuracy, AUC, F1 Score, and MCC. It provided the best overall balance across the evaluation metrics. |
| Decision Tree | Produced lower Accuracy, AUC, F1 Score, and MCC than the leading models, indicating weaker generalization on the test data. |
| K-Nearest Neighbors | Delivered moderate performance, but its Accuracy and MCC were below Logistic Regression, SVM, and Random Forest. |
| Naive Bayes | Achieved the highest Recall, which indicates stronger identification of churn cases, but its lower Precision and Accuracy show that it generated more false-positive predictions. |
| Random Forest | Produced strong Accuracy and AUC and was one of the better-performing models, although it remained below Logistic Regression overall. |
| Support Vector Machine | Achieved the second-highest Accuracy and MCC and provided relatively balanced performance. |
| **Overall Winner** | **Logistic Regression was selected as the best-performing model because it achieved the highest Accuracy, AUC, F1 Score, and MCC among the evaluated models.** |

## 7. Streamlit Application Features

The deployed Streamlit application provides:

- Test CSV dataset upload
- Classification model selection dropdown
- Display of Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- Confusion matrix display
- Overall model comparison table
- Model performance analysis and conclusion

## 8. Logistic Regression Confusion Matrix

The displayed Logistic Regression confusion matrix contains:

| Actual / Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 925 | 110 |
| Churn | 162 | 212 |

The model correctly classified 925 non-churn customers and 212 churn customers. It incorrectly classified 110 non-churn customers as churn and 162 churn customers as non-churn.

## 9. Conclusion

Six machine learning classification models were implemented and compared using the Telco Customer Churn dataset.

Logistic Regression achieved the best overall performance, with an Accuracy of 0.8070, AUC of 0.8416, F1 Score of 0.6092, and MCC of 0.4843. Naive Bayes achieved the highest Recall but had lower Precision and Accuracy. Random Forest and Support Vector Machine also demonstrated competitive performance.

Based on the evaluation results, Logistic Regression was selected as the overall best-performing model for this customer churn classification problem.

## 10. Repository Contents
```text
streamlit-app/
│
├── app.py
├── requirements.txt
├── README.md
├── Telco-Customer-Churn.csv
└── 2025AC05906_ML_Assignment2.ipynb
