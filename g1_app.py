
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Machine Learning Classification Dashboard",
    page_icon="🚢",
    layout="wide"
)
st.title("Machine Learning Classification Dashboard")
# Move this below the page configuration block
st.subheader("Comparative Machine Learning Analysis using Logistic Regression, Decision Tree, Random Forest and XGBoost")

# --------------- 

# Load Titanic Dataset
# --------------------------------------------------

df = sns.load_dataset("titanic")

# Add Serial Number
df.insert(0, "Serial_Number", range(1, len(df)+1))

st.subheader("Raw Dataset")
st.dataframe(df.head())

# --------------------------------------------------
# Target Variable
# --------------------------------------------------

target = "survived"

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------

data = df.copy()

# Drop high missing columns
drop_cols = ['deck', 'embark_town', 'alive', 'class', 'who']

for col in drop_cols:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

# Separate X and y
X = data.drop(columns=[target])
y = data[target]

# Remove Serial Number from predictors
X = X.drop(columns=["Serial_Number"])

# Handle Missing Values
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Label Encoding
le = LabelEncoder()

for col in cat_cols:
    X[col] = le.fit_transform(X[col])

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Create Train/Test Data for Excel
# --------------------------------------------------

train_df = data.loc[X_train.index].copy()
test_df = data.loc[X_test.index].copy()

comparison_results = []

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=5000, random_state=42),

    "Decision Tree":
        DecisionTreeClassifier(max_depth=5, random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss'
        )
}

for name, mdl in models.items():

    mdl.fit(X_train, y_train)

    pred = mdl.predict(X_test)

    prob = mdl.predict_proba(X_test)[:,1]

    comparison_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test,pred),
        "Precision": precision_score(y_test,pred),
        "Recall": recall_score(y_test,pred),
        "F1 Score": f1_score(y_test,pred),
        "ROC AUC": roc_auc_score(y_test,prob)
    })

comparison_df = pd.DataFrame(comparison_results)
#Display:
st.subheader("Model Comparison")

st.dataframe(
    comparison_df.sort_values(
        by="ROC AUC",
        ascending=False
    )
)
best_model = comparison_df.sort_values(
    by="ROC AUC",
    ascending=False
).iloc[0]

st.success(
    f"Best Performing Model: {best_model['Model']} "
    f"(ROC-AUC = {best_model['ROC AUC']:.4f})"
)
# --------------------------------------------------
# Model Selection
# --------------------------------------------------

selected_model = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ]
)
st.sidebar.success(f"Selected Model: {selected_model}")

# --------------------------------------------------
# Model Building
# --------------------------------------------------


if selected_model == "Logistic Regression":

    model = LogisticRegression(
        max_iter=5000,
        random_state=42
    )

elif selected_model == "Decision Tree":

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

elif selected_model == "Random Forest":

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

else:

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
model.fit(X_train, y_train)
# --------------------------------------------------
# Train Predictions
# --------------------------------------------------

train_pred = model.predict(X_train)

train_prob = model.predict_proba(X_train)

train_df = df.loc[X_train.index].copy()

train_df["Actual_Survived"] = y_train.values

train_df["Predicted_Survived"] = train_pred

train_df["Probability_0"] = train_prob[:,0]

train_df["Probability_1"] = train_prob[:,1]

# --------------------------------------------------
# Test Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

test_prob = model.predict_proba(X_test)

test_df = df.loc[X_test.index].copy()

test_df["Actual_Survived"] = y_test.values

test_df["Predicted_Survived"] = y_pred

test_df["Probability_0"] = test_prob[:,0]

test_df["Probability_1"] = test_prob[:,1]

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()
# --------------------------------------------------
# Metrics Calculation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

fpr_value = FP / (FP + TN)

tpr_value = TP / (TP + FN)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

metrics_df = pd.DataFrame({
    "Metric":[
        "Selected Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "FPR",
        "TPR",
        "ROC AUC"
    ],
    "Value":[
        selected_model,
        accuracy,
        precision,
        recall,
        f1,
        fpr_value,
        tpr_value,
        roc_auc
    ]
})
# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_df = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr,
    "Threshold": thresholds
})

if selected_model in [
    "Decision Tree",
    "Random Forest",
    "XGBoost"
]:
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(10)
    )

    st.subheader("Top 10 Important Features")

    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax2.set_title(
        "Feature Importance"
    )

    st.pyplot(fig2)
# --------------------------------------------------
# Display Metrics
# --------------------------------------------------
st.success(f"Model Used : {selected_model}")
st.subheader("Model Evaluation")

st.dataframe(metrics_df)

# --------------------------------------------------
# Display Confusion Matrix
# --------------------------------------------------


st.subheader("Confusion Matrix")

st.dataframe(cm_df)
col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")

st.metric("ROC-AUC", f"{roc_auc:.4f}")
# --------------------------------------------------
# ROC Curve Plot
# --------------------------------------------------

st.subheader("ROC Curve")

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

ax.plot([0,1],[0,1],'--')

ax.set_xlabel("False Positive Rate")

ax.set_ylabel("True Positive Rate")

ax.set_title("ROC Curve")

ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# Export to Excel
# --------------------------------------------------

file_name = "Titanic_Model_Output.xlsx"

with pd.ExcelWriter(
    file_name,
    engine='openpyxl'
) as writer:

    # Sheet 1
    df.to_excel(
        writer,
        sheet_name="Raw Data",
        index=False
    )

    # Sheet 2
    train_df.to_excel(
        writer,
        sheet_name="Train Data",
        index=False
    )

    # Sheet 3
    test_df.to_excel(
        writer,
        sheet_name="Test Data",
        index=False
    )

    # Sheet 4
    cm_df.to_excel(
        writer,
        sheet_name="Confusion Matrix"
    )

    # Sheet 5
    metrics_df.to_excel(
        writer,
        sheet_name="Model Evaluation",
        index=False
    )

    # Sheet 6
    roc_df.to_excel(
        writer,
        sheet_name="ROC_Curve Data",
        index=False
    )
    #Sheet 7
    comparison_df.to_excel(
    writer,
    sheet_name="Model_Comparison",
    index=False
)

# Sheet 8
if selected_model in [
    "Decision Tree",
    "Random Forest",
    "XGBoost"
]:
    importance_df.to_excel(
        writer,
        sheet_name="Feature_Importance",
        index=False
    )   
st.success(
    f"Excel file generated successfully: {file_name}"
)

# Download Button
with open(file_name, "rb") as file:

    st.download_button(
        label="Download Excel File",
        data=file,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
st.subheader("Dataset Summary")

summary_df = pd.DataFrame({
    "Dataset": ["Raw Data", "Train Data", "Test Data"],
    "Rows": [len(df), len(train_df), len(test_df)]
})

st.subheader("Business Insights")

st.info(f"""
Selected Model : {selected_model}

Key Findings

• Female passengers exhibited significantly higher survival rates.

• Passenger class (Pclass) strongly influenced survival probability.

• Fare paid showed positive association with survival.

• Younger passengers generally had better survival outcomes.

• Current ROC-AUC Score = {roc_auc:.4f}, indicating the model's ability to distinguish survivors from non-survivors.
""")