import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

st.title("🎈 project cancer risk")

st.subheader('Raw Data')

# Load the dataset
csv_url = "https://raw.githubusercontent.com/swjk1/CancerPredictAI/main/The_Cancer_data_1500_V2.csv"
try:
    df = pd.read_csv(csv_url)
    st.write("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Debug: Display column names
st.write("Dataset Columns:", df.columns.tolist())

# Verify required columns
required_columns = ['Age', 'Gender', 'Smoking', 'Genetic Risk', 'Physical Activity', 'Alcohol Intake', 'Cancer History', 'Diagnosis', 'Cancer Risk']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"The following required columns are missing in the dataset: {missing_columns}")
    st.stop()

# Preprocess the dataset
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  # Convert Gender to numeric

# Check for missing values in required columns
if df[required_columns].isnull().any().any():
    st.warning("Dataset contains missing values. Filling missing values with defaults.")
    df = df.fillna({
        'Age': df['Age'].median(),
        'Gender': 0,
        'Smoking': 0,
        'Genetic Risk': 5,
        'Physical Activity': 5,
        'Alcohol Intake': 5,
        'Cancer History': 0,
        'Diagnosis': 0,
        'Cancer Risk': 0
    })

# Define features (X) and target (y)
X = df[['Age', 'Gender', 'Smoking', 'Genetic Risk', 'Physical Activity', 'Alcohol Intake', 'Cancer History', 'Diagnosis']]
y = df['Cancer Risk']

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
smoking = st.sidebar.slider("Smoking (0-10)", min_value=0, max_value=10, value=5)
genetic_risk = st.sidebar.slider("Genetic Risk (0-10)", min_value=0, max_value=10, value=5)
physical_activity = st.sidebar.slider("Physical Activity (0-10)", min_value=0, max_value=10, value=5)
alcohol_intake = st.sidebar.slider("Alcohol Intake (0-10)", min_value=0, max_value=10, value=5)
cancer_history = st.sidebar.selectbox("Cancer History", options=["Yes", "No"])
diagnosis = st.sidebar.selectbox("Diagnosis", options=[0, 1])

# Encode user inputs
gender_encoded = 0 if gender == "Male" else 1
cancer_history_encoded = 1 if cancer_history == "Yes" else 0

# Prepare input for prediction
input_data = np.array([[age, gender_encoded, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history_encoded, diagnosis]])

# Predict and display result
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"### Predicted Cancer Risk: {prediction[0]}")
