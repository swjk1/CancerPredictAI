import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = "https://raw.githubusercontent.com/swjk1/CancerPredictAI/main/The_Cancer_data_1500_V2.csv"
df = pd.read_csv(url)

# Preprocessing: Encode categorical features
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Example: Male = 0, Female = 1

# Define feature columns and target variable
X = df[['Age', 'Gender', 'Smoking', 'Genetic Risk', 'Physical Activity', 'Alcohol Intake', 'Cancer History', 'Diagnosis']]
y = df['Cancer Risk']  # Replace with actual target column if different

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
