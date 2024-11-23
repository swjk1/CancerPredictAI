uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset:")
    st.dataframe(new_df)

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
csv_url = "https://raw.githubusercontent.com/swjk1/CancerPredictAI/main/The_Cancer_data_1500_V2.csv"

# Try to read and display the CSV file
try:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_url)
    
    # Display the data in Streamlit
    st.write("### Cancer Data:")
    st.dataframe(df)  # Interactive table
    
    # Optional: Display summary statistics
    st.write("### Summary Statistics:")
    st.write(df.describe())
    
except Exception as e:
    st.error(f"An error occurred while loading the CSV file: {e}")


# Define features (X) and target (y)
X = df[['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']]
y = df['Diagnosis']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit app layout
st.title("Cancer Risk Prediction")

# User input section
st.sidebar.header("Input Patient Data")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
smoking = st.sidebar.selectbox("Smoking", options=["No", "Yes"])
genetic_risk = st.sidebar.slider("Genetic Risk (0-2)", min_value=0, max_value=2, value=1)
physical_activity = st.sidebar.slider("Hours of Physical Activity Per Week (0-10)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
alcohol_intake = st.sidebar.slider("Alcohol Intake (0-5)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
cancer_history = st.sidebar.selectbox("Cancer History", options=["No", "Yes"])

# Encode user inputs
gender_encoded = 1 if gender == "Female" else 0
smoking_encoded = 1 if smoking == "Yes" else 0
cancer_history_encoded = 1 if cancer_history == "Yes" else 0

#Handle potential input errors
if age <= 0 or age > 120:
    st.sidebar.error("Age must be between 1 and 120.")
if bmi <= 0 or bmi > 50:
    st.sidebar.error("BMI must be between 10 and 50.")


# Prepare input for prediction
input_data = np.array([[age, gender_encoded, bmi, smoking_encoded, genetic_risk, physical_activity, alcohol_intake, cancer_history_encoded]])

# Predict and display result
if st.sidebar.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of High Risk (Diagnosis=1)
    prediction_percentage = round(prediction_proba * 100, 2)

    # Classify risk level
    if prediction_percentage < 33:
        risk_level = "Low Risk"
    elif prediction_percentage < 66:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    # Display the results
    st.write(f"### Predicted Cancer Risk: {prediction_percentage}%")
    st.write(f"### Risk Level: {risk_level}")

# Predict with the trained model
y_pred = model.predict(X_test)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Accuracy: {accuracy * 100:.2f}%")

# Display classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, output_dict=True)

st.write("### Classification Report")
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Display key metrics for class 1
precision = report_df.loc["1", "precision"]
recall = report_df.loc["1", "recall"]
f1_score = report_df.loc["1", "f1-score"]

st.write("### Key Metrics for Class 1 (Cancer)")
st.write(f"- **Precision:** {precision:.2f}")
st.write(f"- **Recall:** {recall:.2f}")
st.write(f"- **F1-Score:** {f1_score:.2f}")

# Display confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues', values_format='d')  # Use values_format='.2f' for percentages
st.pyplot(fig)





