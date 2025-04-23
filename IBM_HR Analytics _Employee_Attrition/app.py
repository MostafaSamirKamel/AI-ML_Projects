import streamlit as st
import pandas as pd
from joblib import load

# Load the saved Random Forest model and scaler
rf_model = load('random_forest_model.joblib')
scaler = load('scaler.joblib')

# Column names for the DataFrame
columns = [
    "Age",                
    "JobSatisfaction",   
    "WorkLifeBalance",
    "MonthlyIncome",
    "YearsAtCompany",
    "BusinessTravel_Non-Travel",
    "BusinessTravel_Travel_Frequently",
    "BusinessTravel_Travel_Rarely",
    "Department_Human Resources",
    "Department_Research & Development",
    "Department_Sales"
]

# Initialize an empty DataFrame for new data
new_data = pd.DataFrame(columns=columns)

# Streamlit UI to capture input
st.title('Employee Data Entry')

# Age
Age = st.number_input("Enter Age", min_value=18, max_value=100)

# Job Satisfaction (Use the values you mentioned earlier)
JobSatisfaction = st.selectbox("Select Job Satisfaction", [1, 2, 3, 4])

# Work Life Balance
WorkLifeBalance = st.selectbox("Select Work Life Balance", [1, 2, 3, 4])

# Monthly Income
MonthlyIncome = st.number_input("Enter Monthly Income", min_value=0)

# Years at Company
YearsAtCompany = st.number_input("Enter Years at Company", min_value=0)

# Business Travel
BusinessTravel = st.selectbox("Select Business Travel", ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'])

# Department
Department = st.selectbox("Select Department", ['Sales', 'Research & Development', 'Human Resources'])

# Submit button
if st.button('Submit'):
    # Creating the numerical input list
    new_numerical = [[float(Age), float(JobSatisfaction), float(WorkLifeBalance), float(MonthlyIncome), float(YearsAtCompany)]]

    # Apply the scaler transformation
    new_numerical_scal = scaler.transform(new_numerical)

    # Create a dictionary for categorical data
    new_categorical = {
        "BusinessTravel_Non-Travel": 0,
        "BusinessTravel_Travel_Frequently": 0,
        "BusinessTravel_Travel_Rarely": 0,
        "Department_Human Resources": 0,
        "Department_Research & Development": 0,
        "Department_Sales": 0
    }

    # Handle Business Travel encoding
    if BusinessTravel == 'Travel_Rarely':
        new_categorical["BusinessTravel_Travel_Rarely"] = 1
    elif BusinessTravel == 'Travel_Frequently':
        new_categorical["BusinessTravel_Travel_Frequently"] = 1
    elif BusinessTravel == 'Non-Travel':
        new_categorical["BusinessTravel_Non-Travel"] = 1

    # Handle Department encoding
    if Department == 'Sales':
        new_categorical["Department_Sales"] = 1
    elif Department == 'Research & Development':
        new_categorical["Department_Research & Development"] = 1
    elif Department == 'Human Resources':
        new_categorical["Department_Human Resources"] = 1

    # Combine the scaled numerical data and categorical data
    new_row = new_numerical_scal[0].tolist() + list(new_categorical.values())

    # Append the new row to the DataFrame
    new_data.loc[len(new_data)] = new_row

    # Display the updated DataFrame
    st.write("Updated DataFrame:")
    st.write(new_data)

    # Make predictions using the Random Forest model
    prediction = rf_model.predict([new_row])

    # Display the prediction
    st.write("Prediction from Random Forest Model:")
    st.write(prediction)
