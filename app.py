import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# Application Header
st.title("🌸 Iris Flower Species Predictor")
st.write("""
This interactive web app predicts the species of an **Iris flower** (**Setosa**, **Versicolor**, or **Virginica**) 
based on sepal and petal measurements using a trained **Logistic Regression** model.
""")

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load('iris_model.joblib')

try:
    model = load_model()
    st.success("Trained model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Class Name Mapping
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Sidebar Controls with Expanded Slider Ranges (e.g., 0.0 to 20.0)
st.sidebar.header("Input Flower Measurements (cm)")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 0.0, 20.0, 5.8, step=0.1)
    sepal_width = st.sidebar.slider("Sepal Width", 0.0, 20.0, 3.0, step=0.1)
    petal_length = st.sidebar.slider("Petal Length", 0.0, 20.0, 4.3, step=0.1)
    petal_width = st.sidebar.slider("Petal Width", 0.0, 20.0, 1.3, step=0.1)
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display Selected Measurements
st.subheader("Selected Flower Measurements")
st.dataframe(input_df, use_container_width=True)

# Extract values for validation check
sl = input_df['sepal length (cm)'][0]
sw = input_df['sepal width (cm)'][0]
pl = input_df['petal length (cm)'][0]
pw = input_df['petal width (cm)'][0]

# Prediction Button
if st.button("Predict Species 🚀"):
    # Check if inputs fall outside standard Iris dataset boundaries
    if (sl < 4.0 or sl > 8.0) or (sw < 2.0 or sw > 4.5) or (pl < 1.0 or pl > 7.0) or (pw < 0.1 or pw > 2.5):
        st.warning("This is not related to this project")
    else:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        predicted_species = species_map[prediction]

        st.subheader("Prediction Result")
        st.success(f"The predicted species is **Iris {predicted_species}**! 🎉")

        # Display Probabilities
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Species': ['Setosa', 'Versicolor', 'Virginica'],
            'Probability': [f"{p * 100:.2f}%" for p in probabilities]
        })
        st.table(prob_df)