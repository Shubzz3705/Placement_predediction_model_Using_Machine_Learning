import streamlit as st
import numpy as np
import pickle

# Load the pickled model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

def convert_input(input_data):
    # Gender conversion
    input_data['gender'] = 1 if input_data['gender'].lower() == 'male' else 0

    # Work experience conversion
    input_data['workex'] = 1 if input_data['workex'].lower() == 'yes' else 0

    # MBA Specialisation conversion
    input_data['specialisation'] = 1 if input_data['specialisation'] == 'CT' else 0

    # HSC Stream encoding
    hsc_stream_mapping = {
        'arts': 'dummy_Arts',
        'commerce': 'dummy_Commerce',
        'science': 'dummy_Science'
    }
    hsc_column = hsc_stream_mapping[input_data['hsc_s'].lower()]
    for col in ['dummy_Arts', 'dummy_Commerce', 'dummy_Science']:
        input_data[col] = 1 if col == hsc_column else 0

    # Degree Type encoding
    degree_mapping = {
        'comm&mgmt': 'dummy_Comm&Mgmt',
        'sci&tech': 'dummy_Sci&Tech',
        'others': 'dummy_Others'
    }
    degree_column = degree_mapping[input_data['degree_t'].lower()]
    for col in ['dummy_Comm&Mgmt', 'dummy_Sci&Tech', 'dummy_Others']:
        input_data[col] = 1 if col == degree_column else 0

    return input_data

def predict_status(raw_input, model):
    # Convert and process the input data
    input_data = convert_input(raw_input.copy())

    # Prepare input for model
    feature_cols = [
        'gender', 'ssc_p', 'hsc_p', 'degree_p', 'workex',
        'specialisation', 'PG_per', 'dummy_Arts', 'dummy_Commerce',
        'dummy_Science', 'dummy_Comm&Mgmt', 'dummy_Others', 'dummy_Sci&Tech'
    ]
    input_values = [input_data[col] for col in feature_cols]
    input_array = np.array(input_values).reshape(1, -1)

    # Predict using the model
    return model.predict(input_array)[0]

# Streamlit UI Configuration
st.set_page_config(page_title="Placement Status Predictor", layout="wide")  # set wide layout
st.title('Placement Status Predictor')

# Creating columns for layout
col1, col2, col3 = st.columns([1,1,1])

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    ssc_p = st.slider('SSC Percentage', 0.0, 100.0, 50.0)
    hsc_p = st.slider('HSC Percentage', 0.0, 100.0, 50.0)

with col2:
    hsc_s = st.selectbox('HSC Stream', ['Commerce', 'Science', 'Arts'])
    degree_p = st.slider('Degree Percentage', 0.0, 100.0, 50.0)
    degree_t = st.selectbox('Undergraduate Degree Type', ['Comm&Mgmt', 'Sci&Tech', 'Others'])

with col3:
    workex = st.selectbox('Work Experience', ['Yes', 'No'])
    specialisation = st.selectbox('MBA Specialisation', ['DS', 'CT'])
    PG_per = st.slider('PG Percentage', 0.0, 100.0, 50.0)

input_data = {
    'gender': gender,
    'ssc_p': ssc_p,
    'hsc_p': hsc_p,
    'hsc_s': hsc_s,
    'degree_p': degree_p,
    'degree_t': degree_t,
    'workex': workex,
    'specialisation': specialisation,
    'PG_per': PG_per
}

if st.button('Predict'):
    result = predict_status(input_data, model)
    st.subheader(f'Predicted Status: {"Placed" if result == 1 else "Not Placed"}')