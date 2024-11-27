import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
csv_url = "https://raw.githubusercontent.com/swjk1/CancerPredictAI/main/The_Cancer_data_1500_V2.csv"

st.title("Cancer Risk Assessment Model")
df = pd.read_csv(csv_url)

# Define features (X) and target (y)
X = df[['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']]
y = df['Diagnosis']

# Apply Polynomial Features (example: degree=2, for Age and BMI)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X[['BMI', 'Age']])
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(['BMI', 'Age']))

# Add the polynomial features to the original dataframe
X = pd.concat([X, X_poly_df], axis=1)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Sidebar: BMI Input Section
def calculate_bmi(weight, height, unit_system):
    if unit_system == "Metric":
        # Metric Units: weight in kg, height in meters
        bmi = weight / (height ** 2)
    elif unit_system == "US":
        # US Units: weight in pounds, height in inches
        bmi = 703 * (weight / (height ** 2))
    return bmi
    
# Sidebar: Input Patient Data
st.sidebar.header("Input Patient Data")

# Define BMI Input Option (Manual or Calculator)
bmi_option = st.sidebar.radio("Do you know your BMI?", ("Yes", "No"))

# Initialize bmi variable to avoid reference error
bmi = None

if bmi_option == "Yes":
    bmi = st.sidebar.number_input("Enter your BMI", min_value=0.0, step=0.1)

elif bmi_option == "No":
    unit_system = st.sidebar.radio("Select Unit System", ("Metric", "US"))
    
    if unit_system == "Metric":
        weight = st.sidebar.number_input("Enter your weight (kg)", min_value=1.0, step=0.1, format="%.1f")
        height = st.sidebar.number_input("Enter your height (m)", min_value=0.5, step=0.01, format="%.2f")
    else:
        weight = st.sidebar.number_input("Enter your weight (lbs)", min_value=1.0, step=0.1, format="%.1f")
        height = st.sidebar.number_input("Enter your height (inches)", min_value=10, step=0.1, format="%.1f")
        
    # Calculate BMI when the user presses the button
    if st.sidebar.button("Calculate BMI"):
        if weight > 0 and height > 0:
            bmi = calculate_bmi(weight, height, unit_system)
            st.sidebar.write(f"### Your BMI: {bmi:.2f}")

            # Interpretation of BMI
            if bmi < 18.5:
                st.sidebar.write("You are underweight.")
            elif 18.5 <= bmi < 24.9:
                st.sidebar.write("You have a normal weight.")
            elif 25 <= bmi < 29.9:
                st.sidebar.write("You are overweight.")
            else:
                st.sidebar.write("You are obese.")
        else:
            st.sidebar.error("Please enter valid values for weight and height.")

# Check that bmi is defined before using it in the input_data
if bmi is None:
    st.sidebar.error("Please enter a valid BMI value to proceed.")
else:
# Define user inputs
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
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

# Prepare input for prediction (original input data from the sidebar)
input_data = np.array([[age, gender_encoded, bmi, smoking_encoded, genetic_risk, physical_activity, alcohol_intake, cancer_history_encoded]])

# Convert to DataFrame to use the same feature names as the training data
input_df = pd.DataFrame(input_data, columns=['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'])

# Apply Polynomial Features (same as during training)
input_poly = poly.transform(input_df[['BMI', 'Age']])

# Convert the polynomial features to a DataFrame and concatenate with the original input data
input_poly_df = pd.DataFrame(input_poly, columns=poly.get_feature_names_out(['BMI', 'Age']))
input_data_transformed = pd.concat([input_df, input_poly_df], axis=1)

st.markdown(
    """
    <style>
    /* Increase font size for tab buttons */
    div[class*="stTabs"] button {
        font-size: 80px;
        padding: 10px 20px; /* Adjust padding if needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["Results", "Data", "Reliability"])

# Content for each tab
with tab1:
    # Predict and display result only when the "Predict" button is clicked
    if st.sidebar.button("Predict"):
        # Prepare input for prediction (already collected in the sidebar)
        input_data = np.array([[age, gender_encoded, bmi, smoking_encoded, genetic_risk, physical_activity, alcohol_intake, cancer_history_encoded]])

        # Apply the same polynomial transformation to the input data as we did during training
        input_data_poly = poly.transform(input_data[:, [2, 0]])  # Apply to 'Age' (index 0) and 'BMI' (index 2)
        
        # Combine the original input data with the polynomial features
        input_data_transformed = np.hstack([input_data, input_data_poly])

        # Predict the probability of cancer risk
        prediction_proba = model.predict_proba(input_data_transformed)[0][1]  # Probability of High Risk (Diagnosis=1)
        prediction_percentage = round(prediction_proba * 100, 2)

        # Classify risk level based on the probability
        if prediction_percentage < 33:
            risk_level = "Low Risk"
        elif prediction_percentage < 66:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        # Display the prediction results
        st.markdown(f"### Predicted Cancer Risk: **{prediction_percentage}%**")
        st.markdown(f"### Risk Level: **{risk_level}**")
    else:
        st.markdown("### Click **\"Predict\"** to see results")


with tab2:
    # Try to read and display the CSV file
    try:
        # Load the CSV file into a DataFrame
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

# In tab3 where you evaluate the model's performance on the test set
with tab3:
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy score for the model on the test set
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    # Display classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("### Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Display key metrics for class 1 (Cancer Positive), make sure the label "1" exists in your model
    if "1" in report_df.index:
        precision = report_df.loc["1", "precision"]
        recall = report_df.loc["1", "recall"]
        f1_score = report_df.loc["1", "f1-score"]

        st.write(f"### Class 1 (Cancer Positive) Metrics:")
        st.write(f"Precision: {precision * 100:.2f}%")
        st.write(f"Recall: {recall * 100:.2f}%")
        st.write(f"F1-Score: {f1_score * 100:.2f}%")
    else:
        st.write("### No Class 1 (Cancer Positive) found in the model output")

    # Display confusion matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    # Ensure correct display of confusion matrix labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Blues', values_format='d')  # Use values_format='.2f' for percentages
    st.pyplot(fig)
