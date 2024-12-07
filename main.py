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
    if close_relatives >= 3:
        genetic_risk = 3  # High risk
    elif close_relatives == 2 or remote_relatives >= 3:
        genetic_risk = 2  # Moderate risk
    else:
        genetic_risk = 1  # Low risk

# Encoding inputs
gender_encoded = 1 if gender == "Female" else 0
smoking_encoded = 1 if smoking == "Yes" else 0
cancer_history_encoded = 1 if cancer_history == "Yes" else 0

# Prepare input for prediction
input_data = np.array([[age, gender_encoded, bmi, smoking_encoded, genetic_risk, physical_activity, alcohol_intake, cancer_history_encoded]])

# Convert to DataFrame to use the same feature names as the training data
input_df = pd.DataFrame(input_data, columns=['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'])

# Apply Polynomial Features (same as during training)
input_poly = poly.transform(input_df[['BMI', 'Age']])

# Convert the polynomial features to a DataFrame and concatenate with the original input data
input_poly_df = pd.DataFrame(input_poly, columns=poly.get_feature_names_out(['BMI', 'Age']))
input_data_transformed = pd.concat([input_df, input_poly_df], axis=1)

# Tabs setup
tab1, tab2, tab3 = st.tabs(["Results", "Data", "Reliability"])

# Content for each tab
with tab1:
    # Add a button to trigger the prediction
    if st.sidebar.button("Predict"):
        # Predict the probability of cancer risk
        prediction_proba = model.predict_proba(input_data_transformed)[0][1]
        prediction_percentage = round(prediction_proba * 100, 2)

        # Determine risk level and corresponding advice
        if prediction_percentage < 33:
            risk_level = "Low Risk"
            advice = (
                "Routine check-ups every 2-3 years are sufficient. "
                "Maintain a healthy lifestyle with balanced nutrition and regular exercise."
            )
        elif prediction_percentage < 66:
            risk_level = "Medium Risk"
            advice = (
                "Schedule a check-up annually. Adopt a healthy lifestyle and consider consultations "
                "with a healthcare professional for prevention strategies."
            )
        else:
            risk_level = "High Risk"
            advice = (
                "Consult a healthcare provider immediately for further tests and screenings. "
                "Regular check-ups every 6 months are recommended, along with lifestyle adjustments."
            )

        # Display the prediction results
        st.markdown(f"### Predicted Cancer Risk: **{prediction_percentage}%**")
        st.markdown(f"### Risk Level: **{risk_level}**")
        st.markdown(f"### Advice: {advice}")
    else:
        st.markdown("### Click **\"Predict\"** to see results.")

with tab2:
    st.header("Cancer Data:")
    st.dataframe(df)
    st.header("Summary Statistics:")
    st.write(df.describe())

    st.write("### Histograms for Numeric Columns:")
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) > 0:
        fig, axes = plt.subplots(nrows=(len(numeric_columns) + 1) // 2, ncols=2, figsize=(12, 6))
        axes = axes.flatten()

        for i, column in enumerate(numeric_columns):
            axes[i].hist(df[column], bins=20, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"{column} Histogram")
        st.pyplot(fig)

with tab3:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Accuracy: {accuracy:.2%}")
    st.write("### Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)
