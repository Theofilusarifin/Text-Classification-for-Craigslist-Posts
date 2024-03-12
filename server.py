import streamlit as st
import time

from codes.text_preprocessing import text_preproccessing
from codes.section_preprocessing import section_preprocessing

st.title("Text Classification for Craigslist Posts")

cities = ['bangalore', 'chicago', 'delhi', 'dubai.en', 'frankfurt.en', 'geneva.en', 'hyderabad', 'kolkata.en',
          'london', 'manchester', 'mumbai', 'newyork', 'paris.en', 'seattle', 'singapore', 'zurich.en']
selected_city = st.selectbox('City:', cities, index=None)

sections = ['for-sale', 'housing', 'services', 'community']
selected_section = st.selectbox('Section:', sections, index=None)

input_heading = st.text_area("Please Enter Your Heading:")

if st.button("Predict Category"):
    if selected_city and selected_section and input_heading:  # Check if all inputs are filled
        with st.spinner('Predicting category...'):  # Show loading spinner
            time.sleep(1)  # Simulate loading for 2 seconds
            preprocessed_heading = text_preproccessing(input_heading)
            predicted_category = section_preprocessing(selected_section, preprocessed_heading)

        result = f'Predicted Category : {predicted_category}'
        st.success(result)  # Show success message after loading
    else:
        st.warning("Please fill in all inputs before predicting.")
