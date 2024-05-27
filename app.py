import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Customer Churn Prediction")
st.markdown("Predict which customers are likely to stop using a product or service.")

@st.cache_data
def load_data():
    data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return data

@st.cache_data
def preprocess_data(data):
    data.drop(columns=['customerID'], inplace=True)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
    
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    return data, label_encoders

@st.cache_resource
def train_model(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def plot_metrics(metrics):
    st.write("### Model Performance")
    for metric_name, values in metrics.items():
        st.write(f"#### {metric_name}")
        for model_name, value in values.items():
            st.write(f"{model_name}: {value:.4f}")
        st.write("")

def main():
    data = load_data()
    
    if st.sidebar.checkbox("Show raw data"):
        st.subheader("Telecom Churn Data")
        st.write(data)
    
    data, label_encoders = preprocess_data(data)
    
    X = data.drop(columns=['Churn'])
    y = data['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models = train_model(X_train, y_train)
    
    st.sidebar.subheader("Model Selection and Prediction")
    model_choice = st.sidebar.selectbox("Choose Model", ("Logistic Regression", "Random Forest", "Gradient Boosting"))
    
    if st.sidebar.button("Predict"):
        model = models[model_choice]
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        metrics = {
            "Accuracy": {model_choice: accuracy},
            "Precision": {model_choice: precision},
            "Recall": {model_choice: recall},
            "F1 Score": {model_choice: f1},
            "ROC AUC": {model_choice: roc_auc}
        }
        
        plot_metrics(metrics)
        
        st.subheader(f"Confusion Matrix for {model_choice}")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
        
        st.sidebar.subheader("Save Model")
        if st.sidebar.button("Save"):
            joblib.dump(model, f"{model_choice}_model.pkl")
            st.sidebar.write(f"Model saved as {model_choice}_model.pkl")

if __name__ == '__main__':
    main()
