import streamlit as st
import pandas as pd
import pickle
import joblib

st.title('Predicting Machine Failure')

st.write('Hello world!')

# Load the pre-trained models
rf_model_L = joblib.load('rf_model_L.pkl')
rf_model_M = joblib.load('rf_model_M.pkl')
rf_model_H = joblib.load('rf_model_H.pkl')

# Function to make predictions based on the model and input data
def predict_failure(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit App Title
st.title("Machine Failure Prediction")

# Sidebar for machine group selection
machine_group = st.sidebar.selectbox("Select Machine Group", ["L", "M", "H"])

# User inputs for the features
st.header(f"Input Features for Machine Group {machine_group}")

air_temp = st.number_input("Air Temperature (K)", min_value=200, max_value=400)
process_temp = st.number_input("Process Temperature (K)", min_value=200, max_value=400)
rotational_speed = st.number_input("Rotational Speed (rpm)", min_value=0, max_value=10000)
torque = st.number_input("Torque (Nm)", min_value=0, max_value=100)
tool_wear = st.number_input("Tool Wear (min)", min_value=0, max_value=500)

# Prepare the input data as a DataFrame
input_data = pd.DataFrame([[air_temp, process_temp, rotational_speed, torque, tool_wear]], 
                          columns=["Air_temperature", "Process_temperature", "Rotational_speed", "Torque", "Tool_wear"])

# Select the model based on machine group
if machine_group == 'L':
    model = rf_model_L
elif machine_group == 'M':
    model = rf_model_M
else:
    model = rf_model_H

# Predict when the button is pressed
if st.button("Predict Failure"):
    prediction = predict_failure(model, input_data)
    if prediction[0] == 1:
        st.error("The machine is likely to fail.")
    else:
        st.success("The machine is unlikely to fail.")
