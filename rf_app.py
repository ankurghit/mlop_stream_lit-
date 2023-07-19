
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 04:59:29 2023

@author: Ankur Tiwari
"""

import streamlit as st
import pandas as pd
import pickle

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a dictionary to map ordinal categorical values for 'cut', 'color', and 'clarity'
cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

def encode_features(df):
    df['cut'] = df['cut'].map(cut_map)
    df['color'] = df['color'].map(color_map)
    df['clarity'] = df['clarity'].map(clarity_map)
    return df

def predict_price(carat, cut, color, clarity, depth, table, area):
    input_data = pd.DataFrame({
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depth],
        'table': [table],
        'area': [area]
    })
    input_data = encode_features(input_data)
    return model.predict(input_data)[0]

def main():
    st.title("Diamond Price Prediction App")
    st.write("Enter the diamond features to predict its price:")

    carat = st.slider("Carat", min_value=0.2, max_value=5.0, value=1.0, step=0.01)
    cut = st.selectbox("Cut", ('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'))
    color = st.selectbox("Color", ('J', 'I', 'H', 'G', 'F', 'E', 'D'))
    clarity = st.selectbox("Clarity", ('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'))
    depth = st.number_input("Depth", min_value=43.0, max_value=79.0, value=61.5, step=0.1)
    table = st.number_input("Table", min_value=43.0, max_value=95.0, value=57.0, step=0.1)
    area = st.number_input("Area", min_value=0.0, value=0.0)

if __name__ == "__main__":
    main()

    if st.button("Predict"):
        # Call the predict_price function and display the result
        price = predict_price(carat, cut, color, clarity, depth, table, area)
        st.subheader("Predicted Price")
        st.write(f"${price:,.2f}")

if __name__ == "__main__":
    main()
