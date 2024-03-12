import streamlit as st 

st.title("A Simple Sentiment Analysis WebApp.") 

cities = ['bangalore', 'chicago', 'delhi', 'dubai.en', 'frankfurt.en', 'geneva.en', 'hyderabad', 'kolkata.en', 'london', 'manchester', 'mumbai', 'newyork', 'paris.en', 'seattle', 'singapore', 'zurich.en']
selected_city = st.selectbox('City:', cities, index=None)

sections = ['for-sale', 'housing', 'services', 'community']
selected_section = st.selectbox('Section:', sections, index=None)

text = st.text_area("Please Enter Your Heading:")

if st.button("Predict Category"): 
  result = 'Predicted Category : test'
  st.write(result)