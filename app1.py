import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import load


# Load the dataset
@st.cache_data
def load_data():
    url = "Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    column_names = ['customerID','gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn']
    data = pd.read_csv(url, names=column_names, sep=',\s', engine='python')
    return data

# Load the model
model = load('nn_Tuned.joblib')

# Preprocess the data
data = load_data()

# Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split the data
X = data.drop('Churn_Target', 'costumerID', 'Churn', axis=1)
y = data['Churn_Target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
model = RandomForestClassifier(n_estimators = 1000, min_samples_leaf =10, random_state = 42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Title of the web app
st.title('Churn Prediction')

def user_input_features():
    Gender = st.selectbox('Gender', options=list(label_encoders['gender'].classes_))
    Senior Citizen = st.selectbox('SeniorCitizen', options=list(label_encoders['SeniorCitizen'].classes_))
    Partner = st.selectbox('Partner', options=list(label_encoders['Partner'].classes_))
    Dependents = st.selectbox('Dependents', options=list(label_encoders['Dependents'].classes_))
    MultipleLines = st.selectbox('MultipleLines', options=list(label_encoders['MultipleLines'].classes_))
    InternetService = st.selectbox('InternetService', options=list(label_encoders['InternetService'].classes_))
    OnlineSecurity = st.selectbox('OnlineSecurity', options=list(label_encoders['OnlineSecurity'].classes_))
    OnlineBackup = st.selectbox('OnlineBackup', options=list(label_encoders['OnlineBackup'].classes_))
    DeviceProtection = st.selectbox('DeviceProtection', options=list(label_encoders['DeviceProtection'].classes_))
    TechSupport = st.selectbox('TechSupport', options=list(label_encoders['TechSupport'].classes_))
    StreamingTV = st.selectbox('StreamingTV', options=list(label_encoders['StreamingTV'].classes_))
    StreamingMovies = st.selectbox('StreamingMovies', options=list(label_encoders['StreamingMovies'].classes_))
    Contract = st.selectbox('Contract', options=list(label_encoders['Contract'].classes_))
    PaperlessBilling = st.selectbox('PaperlessBilling', options=list(label_encoders['PaperlessBilling'].classes_))
    PaymentMethod = st.selectbox('PaymentMethod', options=list(label_encoders['PaymentMethod'].classes_))

    data_dict = {
        'age': age,
        'workclass': workclass,
        'education-num': education_num_value,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country,
    }

    features = pd.DataFrame(data_dict, index=[0])
    for column in features.select_dtypes(include=['object']).columns:
        features[column] = label_encoders[column].transform(features[column])
    features = scaler.transform(features)
    return features

input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)
    if prediction == 1:
        st.write('Churn')
    else:
        st.write('Not Churn')

st.write("""
### Model Accuracy
""")
st.write(f'Model Accuracy: {accuracy:.2f}')
