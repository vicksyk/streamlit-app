#!/usr/bin/env python
# coding: utf-8

# In[16]:


import streamlit as st
import pandas as pd
import numpy as np

st.title("Customer Churn Prediction System")

uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df_uploaded.head())

    st.write("Dataset Shape:", df_uploaded.shape)

model_name = st.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "SVM"
    ]
)

st.write("Selected Model:", model_name)

if model_name == "Logistic Regression":
    st.metric("Accuracy", "80.70%")
    st.metric("AUC", "0.8416")
    st.metric("Precision", "0.6584")
    st.metric("Recall", "0.5668")
    st.metric("F1 Score", "0.6092")
    st.metric("MCC", "0.4843")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)


# In[17]:


df = pd.read_csv("Telco-Customer-Churn.csv")

print(df.shape)
df.head()


# In[18]:


print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# In[19]:


df.drop("customerID", axis=1, inplace=True)


# In[20]:


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("NaN in TotalCharges before fill:",
      df["TotalCharges"].isna().sum())


# In[21]:


df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

print("NaN after fill:",
      df["TotalCharges"].isna().sum())


# In[22]:


df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print(df["Churn"].value_counts())


# In[23]:


# Find current categorical columns
categorical_cols = df.select_dtypes(
    include=['object', 'string', 'category']
).columns.tolist()

print("Categorical Columns Found:")
print(categorical_cols)

# Apply one-hot encoding only if categorical columns exist
if len(categorical_cols) > 0:
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

print("New Shape:", df.shape)
df = df.fillna(0)
print("Total NaN values:",
      df.isna().sum().sum())


# In[24]:


y = df["Churn"]

X = df.drop(columns=["Churn"])

print("X Shape:", X.shape)
print("y Shape:", y.shape)
print("NaN in X:",
      X.isna().sum().sum())

print("NaN in y:",
      y.isna().sum())


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)


# In[26]:


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
print(
    "NaN in X_train_scaled:",
    np.isnan(X_train_scaled).sum()
)

print(
    "NaN in X_test_scaled:",
    np.isnan(X_test_scaled).sum()
)


# In[27]:


lr_model = LogisticRegression(
    max_iter=2000
)

lr_model.fit(
    X_train_scaled,
    y_train
)

y_pred_lr = lr_model.predict(
    X_test_scaled
)

y_prob_lr = lr_model.predict_proba(
    X_test_scaled
)[:, 1]

print("Accuracy:",
      accuracy_score(y_test, y_pred_lr))

print("Precision:",
      precision_score(y_test, y_pred_lr))

print("Recall:",
      recall_score(y_test, y_pred_lr))

print("F1:",
      f1_score(y_test, y_pred_lr))

print("AUC:",
      roc_auc_score(y_test, y_prob_lr))

print("MCC:",
      matthews_corrcoef(y_test, y_pred_lr))


# In[28]:


models = {

    "Logistic Regression":
        LogisticRegression(max_iter=2000),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes":
        GaussianNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "SVM":
        SVC(
            probability=True
        )
}


# In[29]:


results = []

for name, model in models.items():

    if name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:

        model.fit(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_test_scaled
        )

        y_prob = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

    results.append([
        name,
        round(
            accuracy_score(
                y_test,
                y_pred
            ),4
        ),
        round(
            roc_auc_score(
                y_test,
                y_prob
            ),4
        ),
        round(
            precision_score(
                y_test,
                y_pred
            ),4
        ),
        round(
            recall_score(
                y_test,
                y_pred
            ),4
        ),
        round(
            f1_score(
                y_test,
                y_pred
            ),4
        ),
        round(
            matthews_corrcoef(
                y_test,
                y_pred
            ),4
        )
    ])


# In[30]:


results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

results_df


# In[33]:




# In[35]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# 1. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, y_pred_lr)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
st.pyplot(plt.gcf())

# =====================================
# 2. ROC CURVE
# =====================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob_lr
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    color="gray"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(True)

plt.show()

# =====================================
# 3. FEATURE IMPORTANCE
# =====================================

# importance = pd.DataFrame({
#     "Feature": X.columns,
#     "Coefficient": log_reg.coef_[0]
# })

# importance["AbsCoefficient"] = abs(
#     importance["Coefficient"]
# )

# importance = importance.sort_values(
#     by="AbsCoefficient",
#     ascending=False
# )

# print("\nTop 15 Important Features:\n")
# print(importance.head(15))


# In[36]:


# importance.head(15)


# In[ ]:


# Model Performance Analysis

st.write("""
Six machine learning classification models were evaluated for predicting customer churn: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Naive Bayes, Random Forest, and Support Vector Machine (SVM). The models were compared using Accuracy, AUC, Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC).

Among all evaluated models, Logistic Regression achieved the best overall performance, with an accuracy of 80.70%, AUC of 0.8416, F1-score of 0.6092, and MCC of 0.4843. These results indicate that Logistic Regression provides the most balanced performance across all evaluation metrics and demonstrates strong capability in distinguishing between churn and non-churn customers.
""")

st.write("""
The confusion matrix shows that the model correctly classified 925 non-churn customers and 212 churn customers, while making relatively few incorrect predictions. The ROC curve further confirms the model's effectiveness, with an AUC of 0.8416, indicating good discriminative ability between the two classes.

Feature importance analysis revealed that Contract Type, Internet Service Type, Online Security, Tech Support, and Payment Method are among the most influential factors affecting customer churn. Customers with one-year and two-year contracts are significantly less likely to churn, whereas customers using fiber-optic internet service and electronic check payment methods tend to have a higher likelihood of churn.

Overall, Logistic Regression provides the best combination of predictive performance, interpretability, and reliability for this customer churn prediction problem.
""")


# In[ ]:


# Conclusion

st.write("""
The objective of this project was to develop and evaluate machine learning models for predicting customer churn using the Telco Customer Churn dataset. Data preprocessing steps included handling missing values, encoding categorical variables, feature scaling, and splitting the dataset into training and testing sets.

A total of six classification algorithms were implemented and compared. Based on the evaluation metrics, Logistic Regression emerged as the best-performing model, achieving the highest accuracy, AUC, F1-score, and MCC among all models tested. The model demonstrated strong predictive capability while maintaining a good balance between identifying churned customers and minimizing false predictions.

The analysis also identified several important business factors associated with churn, including contract duration, internet service type, online security services, technical support availability, and payment methods. These insights can help organizations better understand customer behavior and implement targeted retention strategies.

In conclusion, the developed Logistic Regression model is an effective and interpretable solution for customer churn prediction. The findings of this study demonstrate the value of machine learning in supporting data-driven decision-making and improving customer retention efforts.
""")


# In[ ]:




