import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

# Streamlit UI
st.title('Diabetes Prediction')

# Input fields
preg = st.number_input('Pregnancies', min_value=0, step=1)
glucose = st.number_input('Glucose', min_value=0, step=1)
bp = st.number_input('Blood Pressure', min_value=0, step=1)
st = st.number_input('Skin Thickness', min_value=0, step=1)
insulin = st.number_input('Insulin', min_value=0, step=1)
bmi = st.number_input('BMI', min_value=0.0, step=0.1)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.01)
age = st.number_input('Age', min_value=0, step=1)

# Predict button
if st.button('Predict'):
    # Prepare the data
    data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
    # Make prediction
    prediction = classifier.predict(data)
    # Display result
    st.write(f'Prediction: {"Diabetic" if prediction[0] == 1 else "Not Diabetic"}')
