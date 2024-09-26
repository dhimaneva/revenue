import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a title for your app
st.title('Monthly Revenue Prediction App')

# Create input fields for the features
average_page_load_time = st.number_input('Average Page Load Time (seconds)')
average_product_rating = st.number_input('Average Product Rating (out of 5)')
average_shipping_time = st.number_input('Average Shipping Time (days)')
competitor_price_index = st.number_input('Competitor Price Index')
consumer_confidence_index = st.number_input('Consumer Confidence Index')
# Add input fields for other features from your dataset


# Create a button to predict
if st.button('Predict Monthly Revenue'):
    # Create a DataFrame with the input values
    new_data = pd.DataFrame({
        'average_page_load_time': [average_page_load_time],
        'average_product_rating': [average_product_rating],
        'average_shipping_time': [average_shipping_time],
        'competitor_price_index': [competitor_price_index],
        'consumer_confidence_index': [consumer_confidence_index]
        # Add other features from your dataset
    })

    # Ensure new_data has the same columns as your training data
    # You might need to adjust this based on your data and model
    missing_cols = set(df.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0 

    # Make a prediction
    prediction = model.predict(new_data)[0]

    # Display the prediction
    st.write('Predicted Monthly Revenue:', prediction)


# You can add more sections or features to your app as needed
