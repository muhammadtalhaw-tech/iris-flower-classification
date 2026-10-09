import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ==========================================
# 1. Load the Iris Dataset
# ==========================================
print("--- Step 1: Loading Dataset ---")
iris = load_iris()

# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map integer targets to actual species names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species_name'] = df['species'].map(species_map)

print(df.head())
print("\n")

# ==========================================
# 2. Explore & Preprocess Dataset
# ==========================================
print("--- Step 2: Data Exploration & Preprocessing ---")
print("Dataset Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
duplicates = df.duplicated().sum()
print(f"\nDuplicate Records found: {duplicates}")

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates removed successfully.")

# ==========================================
# 3. Identify Features and Target Variable
# ==========================================
# Features (X): Sepal length, Sepal width, Petal length, Petal width
X = df[iris.feature_names]

# Target (y): Species class (0, 1, 2)
y = df['species']

# ==========================================
# 4. Train-Test Split
# ==========================================
# Split data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# ==========================================
# 5. Train Logistic Regression Model
# ==========================================
print("\n--- Step 3: Model Training ---")
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")

# ==========================================
# 6. Evaluate Model
# ==========================================
print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy * 100:.2f}%\n")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ==========================================
# 7. Test Model with New Flower Measurements (Clean Fix)
# ==========================================
print("--- Step 5: Predicting New Samples ---")

new_samples_df = pd.DataFrame([
    [5.1, 3.5, 1.4, 0.2],  # Expected: setosa
    [6.2, 2.9, 4.3, 1.3],  # Expected: versicolor
    [7.3, 2.9, 6.3, 1.8]   # Expected: virginica
], columns=iris.feature_names)

predictions = model.predict(new_samples_df)

for idx, sample in new_samples_df.iterrows():
    predicted_species = species_map[predictions[idx]]
    print(f"Measurements {sample.values.tolist()} --> Predicted Species: {predicted_species}")
    
# ==========================================
# 8. Save Trained Model
# ==========================================
model_filename = 'iris_model.joblib'
joblib.dump(model, model_filename)
print(f"\nModel saved as '{model_filename}' successfully.")