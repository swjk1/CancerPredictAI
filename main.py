import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
csv_url = "https://raw.githubusercontent.com/swjk1/CancerPredictAI/main/The_Cancer_data_1500_V2.csv"
df = pd.read_csv(csv_url)

# Define features (X) and target (y)
X = df[['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']]
y = df['Diagnosis']

# Encode categorical variables
X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})
X['Smoking'] = X['Smoking'].map({'No': 0, 'Yes': 1})
X['CancerHistory'] = X['CancerHistory'].map({'No': 0, 'Yes': 1})

# Polynomial Feature Expansion
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']])  # Polynomial features for numeric columns
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake']))
X = X.drop(['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake'], axis=1)  # Remove original columns
X = pd.concat([X, X_poly_df], axis=1)  # Concatenate the polynomial features

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Define the RandomForest model
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using RandomizedSearchCV
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Best parameters from the random search
best_model = random_search.best_estimator_

# Streamlit App Layout
st.title("Cancer Risk Prediction Model")

# Sidebar: User Input for Prediction
st.sidebar.header("Input Patient Data")

# User input fields
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
smoking = st.sidebar.selectbox("Smoking", options=["No", "Yes"])
cancer_history = st.sidebar.selectbox("Cancer History", options=["No", "Yes"])
physical_activity = st.sidebar.slider("Physical Activity Per Week (hours)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
alcohol_intake = st.sidebar.slider("Alcohol Intake (0-5)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)

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

# Encoding inputs for prediction
gender_encoded = 1 if gender == "Female" else 0
smoking_encoded = 1 if smoking == "Yes" else 0
cancer_history_encoded = 1 if cancer_history == "Yes" else 0

# Adjust genetic risk weight (scaled to 10% influence)
genetic_risk_scaled = genetic_risk * 0.1

# Prepare input data for prediction
input_data = np.array([[age, gender_encoded, bmi, smoking_encoded, genetic_risk_scaled, physical_activity, alcohol_intake, cancer_history_encoded]])

# Polynomial features for the input data
input_data_poly = poly.transform(input_data[:, [0, 2, 5, 6]])  # Only transform numeric features
input_data = np.concatenate([input_data[:, [0, 1, 3, 4, 7]], input_data_poly], axis=1)

# Streamlit tabs
tab1, tab2, tab3 = st.tabs(["Results", "Data", "Reliability"])

with tab1:
    # Predict and display the result
    if st.sidebar.button("Predict"):
        prediction_proba = best_model.predict_proba(input_data)[0][1]  # Probability of High Risk (Diagnosis=1)
        prediction_percentage = round(prediction_proba * 100, 2)

        # Classify risk level
        if prediction_percentage < 33:
            risk_level = "Low Risk"
        elif prediction_percentage < 66:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        # Display the results
        st.markdown(f"### Predicted Cancer Risk: **{prediction_percentage}%**")
        st.markdown(f"### Risk Level: **{risk_level}**")
    else:
        st.markdown("### Click **\"Predict\"** to see results")

with tab2:
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

with tab3:
    # Accuracy and Classification Metrics
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Accuracy: {accuracy * 100:.2f}%")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("### Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    st.pyplot(fig)
