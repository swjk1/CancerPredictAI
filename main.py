import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load the dataset
csv_url = "https://raw.githubusercontent.com/swjk1/CancerPredictAI/main/The_Cancer_data_1500_V2.csv"

st.title("Cancer Risk Assessment Model")

# Try to read and display the CSV file
try:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_url)
    
    # Display the data in Streamlit
    st.header("Cancer Data:")
    st.dataframe(df)  # Interactive table
    
    # Optional: Display summary statistics
    st.header("Summary Statistics:")
    st.write(df.describe())
    
    # Automatically generate histograms for all numeric columns
    st.write("### Histograms for Numeric Columns:")

    # Select numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    num_cols = len(numeric_columns)

    if num_cols > 0:
        # Create subplots for all numeric columns
        fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(12, 4 * ((num_cols + 1) // 2)))
        axes = axes.flatten()  # Flatten axes for easy iteration
    
        for i, column in enumerate(numeric_columns):
            ax = axes[i]
            ax.hist(df[column], bins=20, color='darkblue', alpha=0.7, edgecolor='black')
            ax.set_title(f"Histogram of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            ax.grid(True, linestyle='--', alpha=0.7)
    
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
    
        # Display the histograms
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for histogram generation.")

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

# Sidebar: Input Patient Data
st.sidebar.header("Input Patient Data")

# Define user inputs
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
smoking = st.sidebar.selectbox("Smoking", options=["No", "Yes"])
cancer_history = st.sidebar.selectbox("Cancer History", options=["No", "Yes"])
physical_activity = st.sidebar.slider(
    "Hours of Physical Activity Per Week (0-10)", min_value=0.0, max_value=10.0, value=5.0, step=0.1
)
alcohol_intake = st.sidebar.slider(
    "Alcohol Intake (0-5)", min_value=0.0, max_value=5.0, value=2.5, step=0.1
)

# Sidebar: Genetic Risk Assessment
family_history = st.sidebar.selectbox("Do you have a family history of cancer?", ["No", "Yes"])

if family_history == "No":
    genetic_risk = 0
else:
    close_relatives = st.sidebar.slider("How many close relatives (parents, siblings, children) have been diagnosed with cancer?", 0, 10, 0)
    remote_relatives = st.sidebar.slider("How many remote relatives (grandparents, uncles, aunts, cousins) have been diagnosed with cancer?", 0, 10, 0)
    early_diagnosis = st.sidebar.selectbox("Were any of these diagnoses at an early age (below 50)?", ["No", "Yes"])
    if close_relatives >= 2 or early_diagnosis == "Yes":
        genetic_risk = 2
    elif close_relatives == 1 or remote_relatives >= 2:
        genetic_risk = 1
    else:
        genetic_risk = 0

# Encoding inputs
gender_encoded = 1 if gender == "Female" else 0
smoking_encoded = 1 if smoking == "Yes" else 0
cancer_history_encoded = 1 if cancer_history == "Yes" else 0

# Prepare input for prediction
input_data = np.array([[age, gender_encoded, bmi, smoking_encoded, genetic_risk, physical_activity, alcohol_intake, cancer_history_encoded]])

# Handle potential input errors
if age <= 0 or age > 120:
    st.sidebar.error("Age must be between 1 and 120.")
if bmi <= 0 or bmi > 50:
    st.sidebar.error("BMI must be between 10 and 50.")

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
    st.write(f"### Predicted Cancer Risk: **{prediction_percentage}%**")
    st.write(f"### Risk Level: **{risk_level}**")
    st.write("")

# Predict with the trained model
y_pred = model.predict(X_test)

# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Accuracy: {accuracy * 100:.2f}%")

# Display classification report

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



st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='Blues', values_format='d')  # Use values_format='.2f' for percentages
st.pyplot(fig)




tab1, tab2, tab3 = st.tabs(["Home", "Data", "Settings"])

# Content for each tab
with tab1:
    st.header("Welcome to the Home Tab")
    st.write("This is the content for the Home tab.")

with tab2:
    st.header("Data Tab")
    st.write("Here you can display your data.")
    st.write({"key": "value"})  # Example of data display

with tab3:
    st.header("Settings Tab")
    st.write("Customize your settings here.")
    option = st.selectbox("Choose an option:", ["Option 1", "Option 2", "Option 3"])
    st.write(f"You selected: {option}")
