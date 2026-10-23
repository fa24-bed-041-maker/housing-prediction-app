import streamlit as st
import joblib
import numpy as np

# Load saved model
model = joblib.load('best_housing_model.pkl')

st.title("California House Price Prediction App")
st.write("Enter neighborhood values to predict Median House Value ($100k):")

med_inc = st.number_input("Median Income (in $10k)", value=3.5)
house_age = st.number_input("Median House Age", value=25.0)
ave_rooms = st.number_input("Average Rooms", value=5.0)
ave_bedrms = st.number_input("Average Bedrooms", value=1.0)
population = st.number_input("Population", value=1000.0)
ave_occup = st.number_input("Average Occupancy", value=3.0)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)

if st.button("Predict Price"):
    features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    prediction = model.predict(features)
    st.success(f"Estimated Median House Value: ${prediction[0]*100000:,.2f}")
