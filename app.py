import streamlit as st
import pandas as pd
import mlflow.pyfunc

# Load the model from the MLflow model registry
model_name = "BMI_KNN_Model"
model_version = 1  # Adjust the version as necessary
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# Create the UI
st.title('BMI Prediction')

# Input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
height = st.number_input('Height (in cm)', min_value=130, max_value=200, value=130)
weight = st.number_input('Weight (in kg)', min_value=30, max_value=150, value=30)

# Map gender to numerical values
gender_map = {'Male': 0, 'Female': 1}
gender = gender_map[gender]

# Dictionary to map the prediction to labels
bmi_labels = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

# Predict BMI Index
if st.button('Predict BMI'):
    # Validation checks
    if height < 130 or height > 200:
        st.error('Height must be between 130 and 200 cm.')
    elif weight < 30 or weight > 150:
        st.error('Weight must be between 30 and 150 kg.')
    else:
        input_data = pd.DataFrame([[gender, height, weight]], columns=['Gender', 'Height', 'Weight'])
        prediction = model.predict(input_data)[0]
        prediction_label = bmi_labels.get(prediction, "Unknown")
        st.write(f'Predicted BMI Index: {prediction_label}')
