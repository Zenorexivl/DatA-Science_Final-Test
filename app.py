import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Churn Prediction",
                   layout="wide",
                   page_icon="🏃‍♂️🏃‍♀️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

Churn_model = pickle.load(open(f'{working_dir}/copy_of_churn_probability_prediction.py', 'rb'))

    # page title
    st.title('Churn Prediction')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.text_input('Gender')

    with col2:
        SeniorCitizen = st.text_input('Senior Citizen')

    with col3:
        Dependents = st.text_input('Dependents')

    with col1:
        PhoneService = st.text_input('PhoneS ervice')

    with col2:
        InternetService = st.text_input('Internet Service')

    with col3:
        OnlineSecurity = st.text_input('Online Security')

    with col1:
        OnlineBackup = st.text_input('Online Backup')

    with col2:
        DeviceProtection = st.text_input('DeviceProtection')


    # code for Prediction
    Chrn_diagnosis = ''

    # creating a button for Prediction

    if st.button('Churn Test Result'):

        user_input = [Gender, SeniorCitizen, Dependents, PhoneService, InternetService,
                      OnlineSecurity, OnlineBackup, DeviceProtection]

        user_input = [float(x) for x in user_input]

       Churn_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)
