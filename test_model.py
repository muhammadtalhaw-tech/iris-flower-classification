import pandas as pd
import joblib

# Load trained model
model = joblib.load('iris_model.joblib')
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Test dataset
test_samples = [
    {"name": "Test 1 (Setosa)", "data": [5.1, 3.5, 1.4, 0.2]},
    {"name": "Test 2 (Versicolor)", "data": [6.2, 2.9, 4.3, 1.3]},
    {"name": "Test 3 (Virginica)", "data": [7.3, 2.9, 6.3, 1.8]},
    {"name": "Test 4 (Borderline)", "data": [5.9, 3.0, 4.2, 1.5]},
    {"name": "Test 5 (Invalid Range)", "data": [12.0, 1.0, 15.0, 0.0]}
]

print("--- STARTING AUTOMATED TESTS ---\n")

for test in test_samples:
    sl, sw, pl, pw = test["data"]
    
    # Input range validation check
    if (sl < 4.0 or sl > 8.0) or (sw < 2.0 or sw > 4.5) or (pl < 1.0 or pl > 7.0) or (pw < 0.1 or pw > 2.5):
        print(f"{test['name']}: {test['data']} --> Output: This is not related to this project")
    else:
        df = pd.DataFrame([test["data"]], columns=feature_names)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][pred] * 100
        print(f"{test['name']}: {test['data']} --> Predicted: Iris {species_map[pred]} ({prob:.1f}% confidence)")

print("\n--- ALL TESTS COMPLETED ---")