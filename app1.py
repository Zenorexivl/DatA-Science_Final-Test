import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    data = pd.read_csv(url, names=column_names, sep=',\s', engine='python')
    return data

# Preprocess the data
data = load_data()

# Remove 'education' and 'fnlwgt' columns
data = data.drop(['education', 'fnlwgt'], axis=1)

# Replace '?' with appropriate values
data['workclass'] = data['workclass'].replace('?', 'Private')
data['occupation'] = data['occupation'].replace('?', 'Prof-speciality')
data['native-country'] = data['native-country'].replace('?', 'United-States')

# Combine 'Never-worked' and 'Without-pay' into 'Without pay'
data['workclass'] = data['workclass'].replace(['Never-worked', 'Without-pay'], 'Without pay')

# Combine 'State-gov' and 'Local-gov' into 'Government'
data['workclass'] = data['workclass'].replace(['State-gov', 'Local-gov'], 'Government')

# Rename workclass categories
data['workclass'] = data['workclass'].replace({
    'Federal-gov': 'Federal Government',
    'Self-emp-inc': 'Self Employed'
})

# Include 'Self-emp-not-inc' in 'Private' category
data['workclass'] = data['workclass'].replace('Self-emp-not-inc', 'Private')

# Combine certain marital statuses into 'No spouse'
data['marital-status'] = data['marital-status'].replace(['Divorced', 'Married-spouse-absent', 'Separated', 'Widowed', 'Married-AF-spouse'], 'No spouse')

# Rename marital status categories
data['marital-status'] = data['marital-status'].replace({
    'Married-civ-spouse': 'Married',
    'Never-married': 'Never married'
})

# Combine 'Not-in-family', 'Own-child', 'Unmarried', and 'Other-relative' into 'Other'
data['relationship'] = data['relationship'].replace(['Not-in-family', 'Own-child', 'Unmarried', 'Other-relative'], 'Other')

# Combine 'Amer-Indian-Eskimo' and 'Other' into 'Others'
data['race'] = data['race'].replace(['Amer-Indian-Eskimo', 'Other'], 'Others')

# Rename race category
data['race'] = data['race'].replace('Asian-Pac-Islander', 'Asian')

# Rename native country categories
data['native-country'] = data['native-country'].replace({
    'Hong': 'Hongkong',
    'Outlying-US(Guam-USVI-etc)': 'Other',
    'Holand-Netherlands': 'Netherlands'
})

# Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split the data
X = data.drop('income', axis=1)
y = data['income']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Education number mapping
education_num_mapping = {
    1: "Preschool",
    2: "Elementary School",
    3: "Elementary School",
    4: "Junior High School",
    5: "Junior High School",
    6: "Senior High School",
    7: "Senior High School",
    8: "Senior High School",
    9: "High school graduate",
    10: "College",
    11: "Vocational Edu.",
    12: "Associate Academy",
    13: "Bachelor Degree",
    14: "Master Degree",
    15: "Professional School",
    16: "Doctoral"
}

# Get unique education levels sorted by their corresponding numbers
unique_education_levels = sorted(set(education_num_mapping.values()), key=lambda x: list(education_num_mapping.values()).index(x))

# Streamlit app
st.title("Income Prediction App")

def user_input_features():
    age = st.number_input('Age', min_value=0, max_value=100, value=25)
    workclass = st.selectbox('Workclass', options=list(label_encoders['workclass'].classes_))
    education_num = st.selectbox('Education Level', options=unique_education_levels)
    marital_status = st.selectbox('Marital Status', options=list(label_encoders['marital-status'].classes_))
    occupation = st.selectbox('Occupation', options=list(label_encoders['occupation'].classes_))
    relationship = st.selectbox('Relationship', options=['Husband', 'Wife'] + sorted(set(label_encoders['relationship'].classes_) - {'Husband', 'Wife'}))
    race = st.selectbox('Race', options=list(label_encoders['race'].classes_))
    sex = st.selectbox('Sex', options=['Male', 'Female'])
    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=100000, value=0)
    hours_per_week = st.number_input('Hours per Week', min_value=0, max_value=100, value=40)
    native_country_options = list(label_encoders['native-country'].classes_)
    native_country_options.sort(key=lambda x: (x != 'Other', x))
    native_country = st.selectbox('Native Country', options=native_country_options)

    # Reverse map the selected education level to the corresponding number
    education_num_rev_mapping = {v: k for k, v in education_num_mapping.items()}
    education_num_value = education_num_rev_mapping[education_num]

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
        st.write('Income is >50K')
    else:
        st.write('Income is <=50K')

st.write("""
### Model Accuracy
""")
st.write(f'Model Accuracy: {accuracy:.2f}')
