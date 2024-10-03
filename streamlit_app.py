import streamlit as st
import joblib
import pandas as pd

st.title('🎈 App Name')

st.write('Hello world!')

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
@st.cache  # Cache the function to avoid re-running on every interaction
def load_data():
    filepath = './machine failure.csv'
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        'Air temperature [K]': 'Air_temperature',
        'Process temperature [K]': 'Process_temperature',
        'Rotational speed [rpm]': 'Rotational_speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool_wear'
    })
    return df

# Function to train Random Forest model for each machine group
def train_model(df, machine_group):
    # Filter the data for the selected machine group
    df_group = df[df['Type'] == machine_group]
    
    # Define features and target
    X = df_group[['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear']]
    y = df_group['Machine failure']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)
    
    return rf

# Function to make predictions
def predict_failure(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app
st.title("Machine Failure Prediction App")

# Load the dataset
df = load_data()

# Sidebar for machine group selection
machine_group = st.sidebar.selectbox("Select Machine Group", ["L", "M", "H"])

# Train the model based on the selected machine group
model = train_model(df, machine_group)

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

# Predict failure when the button is clicked
if st.button("Predict Failure"):
    prediction = predict_failure(model, input_data)
    if prediction[0] == 1:
        st.error("The machine is likely to fail.")
    else:
        st.success("The machine is unlikely to fail.")
