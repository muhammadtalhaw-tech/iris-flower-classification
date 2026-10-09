import pandas as pd
import joblib

# Saved model aur label mapping load karein
model = joblib.load('iris_model.joblib')
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Specific feature names jin par model train hua tha
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Custom flower values
user_input = pd.DataFrame([
    [5.9, 3.0, 5.1, 1.8],  # Custom flower 1
    [4.8, 3.4, 1.6, 0.2]   # Custom flower 2
], columns=feature_names)

# Prediction
predictions = model.predict(user_input)

print("--- Standalone Model Prediction ---")
for idx, pred in enumerate(predictions):
    print(f"Sample {idx + 1}: {species_map[pred]}")