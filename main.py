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

# File uploader for testing other datasets
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    try:
        # Read the uploaded file into a DataFrame
        new_df = pd.read_csv(uploaded_file)

        st.write("### Uploaded Dataset:")
        st.dataframe(new_df)

        # Ensure the dataset has the required columns
        required_columns = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory', 'Diagnosis']
        if not all(col in new_df.columns for col in required_columns):
            st.error(f"The uploaded dataset must contain the following columns: {required_columns}")
        else:
            # If dataset is valid, extract features and target
            X_new = new_df[['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']]
            y_new = new_df['Diagnosis']

            # Use the pre-trained model to make predictions on the new dataset
            y_new_pred = model.predict(X_new)

            # Evaluate the new dataset
            from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

            st.write("### Classification Report on Uploaded Dataset")
            report_new = classification_report(y_new, y_new_pred, output_dict=True)
            report_df_new = pd.DataFrame(report_new).transpose()
            st.dataframe(report_df_new)

            st.write("### Confusion Matrix for Uploaded Dataset")
            cm_new = confusion_matrix(y_new, y_new_pred)
            disp_new = ConfusionMatrixDisplay(confusion_matrix=cm_new, display_labels=model.classes_)
            fig_new, ax_new = plt.subplots()
            disp_new.plot(ax=ax_new)
            st.pyplot(fig_new)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a dataset to test the model.")

# Streamlit app layout
st.title("Cancer Risk Prediction")

# User input section
st.sidebar.header("Input Patient Data")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
smoking = st.sidebar.selectbox("Smoking", options=["No", "Yes"])

# Sidebar: Genetic Risk Assessment
st.sidebar.subheader("Genetic Risk Assessment")

# Step 1: Do they have a family history of cancer?
family_history = st.sidebar.selectbox("Do you have a family history of cancer?", ["No", "Yes"])

if family_history == "No":
    genetic_risk = 0  # No family history, low risk
else:
    # Step 2: Ask about close relatives
    close_relatives = st.sidebar.slider(
        "How many close relatives (parents, siblings, children) have been diagnosed with cancer?",
        0, 10, 0
    )

    # Step 3: Ask about remote relatives
    remote_relatives = st.sidebar.slider(
        "How many remote relatives (grandparents, uncles, aunts, cousins) have been diagnosed with cancer?",
        0, 10, 0
    )

    # Step 4: Was early diagnosis involved?
    early_diagnosis = st.sidebar.selectbox(
        "Were any of these diagnoses at an early age (below 50)?", ["No", "Yes"]
    )

    # Calculate genetic risk based on responses
    if close_relatives >= 2 or early_diagnosis == "Yes":
        genetic_risk = 2  # High genetic risk
    elif close_relatives == 1 or remote_relatives >= 2:
        genetic_risk = 1  # Medium genetic risk
    else:
        genetic_risk = 0  # Low genetic risk

# Encode user inputs (use genetic_risk for prediction)
gender_encoded = 1 if gender == "Female" else 0
smoking_encoded = 1 if smoking == "Yes" else 0
cancer_history_encoded = 1 if cancer_history == "Yes" else 0

# Prepare input for prediction
input_data = np.array([[age, gender_encoded, bmi, smoking_encoded, genet


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

#Visualize the distribution of Breast Cancer Patient used as sample in this model
#Goal: identify patterns and potential outliers in the data
if st.checkbox("Histograms of Cancer Patient Data Distribution"):
    st.sidebar.header("Histograms of Cancer Patient Data Distribution")
    selected_column = st.sidebar.selectbox("Select a column for histogram:", options=df.columns)
    if st.sidebar.button("Show Histogram"):
        plt.figure(figsize=(10, 6))
        plt.hist(df[selected_column], bins=20, color='blue', alpha=0.7, edgecolor='black')
        plt.title(f"Histogram of {selected_column}")
        plt.xlabel(selected_column)
        plt.ylabel("Frequency")
        st.pyplot(plt)
    if st.checkbox("Show Histograms for All Columns"):
        st.write("Histograms for All Columns")
        for column in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df[column], bins=20, color='dark blue', alpha=0.7, edgecolor='black')
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            st.pyplot(plt)






