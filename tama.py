import streamlit as st
import pickle
import numpy as np

# Load the machine learning model
model = pickle.load(open('tama.pkl', 'rb'))

# Load the StandardScaler
scaler = pickle.load(open('scale.pkl', 'rb'))

def predict_weather(MinTemp, MaxTemp, Rainfall, WindGustSpeed, Humidity9am, Humidity3am, Pressure9am, Pressure3am):
    # Create a numpy array with the feature values
    features = np.array([[MinTemp, MaxTemp, Rainfall, WindGustSpeed, Humidity9am, Humidity3am, Pressure9am, Pressure3am]])
    
    # Scale the features using the StandardScaler
    scaled_features = scaler.transform(features)
    
    # Make the prediction
    prediction = model.predict(scaled_features)
    
    return prediction[0]

def main():
    st.title('Tama Weather Predictor')
    
    # Get the feature values from the user
    MinTemp = st.number_input('Minimum Temperature')
    MaxTemp = st.number_input('Maximum Temperature')
    Rainfall = st.number_input('Rainfall')
    WindGustSpeed = st.number_input('Wind Gust Speed')
    Humidity9am = st.number_input('Humidity 9am')
    Humidity3am = st.number_input('Humidity 3am')
    Pressure9am = st.number_input('Pressure 9am')
    Pressure3am = st.number_input('Pressure 3am')
    
    # Check if the user clicked the "Predict" button
    if st.button('Predict'):
        # Call the predict_weather function to get the prediction
        prediction = predict_weather(MinTemp, MaxTemp, Rainfall, WindGustSpeed, Humidity9am, Humidity3am, Pressure9am, Pressure3am)
        
        # Convert the prediction to string
        if prediction == 1:
            result = 'Yes'
        else:
            result = 'No'
        
        # Display the prediction result
        st.write('Prediction:', result)

if __name__ == '__main__':
    main()
