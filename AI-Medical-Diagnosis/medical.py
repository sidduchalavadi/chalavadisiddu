# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Title for the Streamlit app
st.title("AI-Powered Medical Diagnosis System")

# Load dataset ( Indians Diabetes dataset as an example)
url = "diabetes_data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# Show the dataset in the app
st.write("### Dataset Preview")
st.dataframe(data.head())

# Data preprocessing
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar for model selection
st.sidebar.title("Choose Model")
model_type = st.sidebar.selectbox("Select the machine learning model", 
                                  ("Random Forest","SVM"))

# Model selection based on user choice
if model_type == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = SVC(probability=True, random_state=42)

# Train the selected model
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)


# ROC curve and AUC
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # For ROC AUC curve
roc_auc = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC Curve
st.write(f"### {model_type} ROC Curve (AUC = {roc_auc:.4f})")
plt.figure(figsize=(6, 3))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{model_type} ROC Curve')
plt.legend()
st.pyplot(plt)

# User input section for making predictions
st.write("### Enter Patient Data for Prediction")
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Making prediction for new data
input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)

# Display the prediction result
if prediction[0] == 1:
    st.write("### The patient is likely to have diabetes.")
else:
    st.write("### The patient is not likely to have diabetes.")
