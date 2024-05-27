# Customer Churn Prediction

This project aims to predict customer churn using data from a telecom company. The application is built using Python and Streamlit, and it leverages various machine learning algorithms to predict which customers are likely to stop using the service.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Deployment](#model-deployment)
- [Acknowledgements](#acknowledgements)

## Project Overview

Customer churn prediction models are used to identify customers who are likely to cancel a service or subscription. This project uses data from the telecom sector to build a predictive model that helps identify churn risk. The model can be used by telecom companies to retain customers by taking proactive measures.

## Features

- Data Upload and Display
- Data Preprocessing
- Model Training (Logistic Regression, Random Forest, Gradient Boosting)
- Model Evaluation (Accuracy, Precision, Recall, F1 Score, ROC AUC)
- Confusion Matrix Visualization
- Model Saving

## Requirements

- Python 3.7 or higher
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Joblib

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Piyushpalasiya/Piyush-Telco-Customer-Churn-project.git
   cd customer-churn-prediction

2. **Create and Activate a Virtual Environment**
   python -m venv myenv
   myenv\Scripts\activate   # On Windows
   source myenv/bin/activate  # On macOS/Linux

3. **Install Dependencies**
    python -m venv myenv
    myenv\Scripts\activate   # On Windows

4. **Run the Streamlit Application**
     streamlit run app.py

  

# Project Structure

customer-churn-prediction/
│
├── app.py                   # Main application file
├── telecom_churn.csv        # Dataset file
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation


## Model Training
 **The application trains three machine learning models:**

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
**These models are trained on the preprocessed data. The preprocessing includes handling missing values, encoding categorical variables, and scaling numerical features.**



# Model Evaluation
**The application evaluates the trained models using the following metrics:**

1.Accuracy
2.Precision
3.Recall
4.F1 Score
5.ROC AUC
**Additionally, it displays a confusion matrix for a visual representation of the model performance.**

# Model Deployment
The best-performing model can be saved and used for future predictions. The model is saved using Joblib, and it can be loaded later for making predictions on new data.

