import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title("🎈 project cancer risk")

st.subheader('Raw Data')

csv_url = "https://raw.githubusercontent.com/swjk1/CancerPredictAI/main/The_Cancer_data_1500_V2.csv"

# Try to read and display the CSV file
try:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_url)
    
    # Display the data in Streamlit
    st.write("### Here is the Cancer Data:")
    st.dataframe(df)  # Interactive table
    
    # Optional: Display summary statistics
    st.write("### Summary Statistics:")
    st.write(df.describe())
    
except Exception as e:
    st.error(f"An error occurred while loading the CSV file: {e}")

# Load the dataset and preprocess
url = "https://raw.githubusercontent.com/swjk1/CancerPredictAI/main/The_Cancer_data_1500_V2.csv"
df = pd.read_csv(url)

# Preprocess: Encode categorical variables (example: Gender)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Example: Male = 0, Female = 1

# Define features and target
X = df[['Age', 'Gender', 'Smoking', 'Genetic Risk', 'Physical Activity', 'Alcohol Intake', 'Cancer History', 'Diagnosis']]
y = df['Cancer Risk']  # Replace with actual target column if different

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Streamlit app
st.title("Cancer Risk Prediction")

# User input
st.sidebar.header("Input Patient Data")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=['Male', 'Female'])
smoking = st.sidebar.slider("Smoking (1-10)", min_value=0, max_value=10, value=5)
genetic_risk = st.sidebar.slider("Genetic Risk (1-10)", min_value=0, max_value=10, value=5)
physical_activity = st.sidebar.slider("Physical Activity (1-10)", min_value=0, max_value=10, value=5)
alcohol_intake = st.sidebar.slider("Alcohol Intake (1-10)", min_value=0, max_value=10, value=5)
cancer_history = st.sidebar.selectbox("Cancer History", options=['Yes', 'No'])
diagnosis = st.sidebar.selectbox("Diagnosis (0 or 1)", options=[0, 1])

# Encode categorical features
gender_encoded = 0 if gender == 'Male' else 1
cancer_history_encoded = 1 if cancer_history == 'Yes' else 0

# Prepare input for prediction
input_data = np.array([[age, gender_encoded, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history_encoded, diagnosis]])

# Predict cancer risk
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"### Predicted Cancer Risk: {prediction[0]}")
