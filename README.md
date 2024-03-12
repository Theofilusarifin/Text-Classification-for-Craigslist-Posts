# Text Classification for Craigslist Posts

This project aims to classify Craigslist posts into different categories based on their heading. It utilizes machine learning models to predict the category of a given heading within a selected city and section.

## Usage

### Running the Streamlit App

1. Run the Streamlit app by executing the following command:

    ```bash
    streamlit run app.py
    ```

2. Select a city and section from the dropdown menus.
3. Enter the heading of the Craigslist post.
4. Click the "Predict Category" button to see the predicted category.

## Preprocessing

The project involves several preprocessing steps on the input text:

1. **Tokenization**: Splitting the text into individual words or tokens.
   This step breaks down the text into smaller units for further analysis.

2. **Removing special characters and URLs**: Eliminating URLs and non-alphanumeric characters from the text.
   Special characters and URLs are often noise in the text data and need to be removed for better analysis.

3. **Removing numeric characters**: Eliminating numerical digits from the text.
   Numeric characters might not contribute much to the meaning of the text and can be safely removed.

4. **Removing emoticons and emojis**: Stripping emoticons and emojis from the text.
   Emoticons and emojis might not provide useful information for text classification and can be removed.

5. **Stemming and lemmatization**: Reducing words to their base or root form to normalize text.
   This step ensures that different forms of the same word are treated as the same token.

6. **Removing stopwords**: Filtering out common words that do not contribute much to the meaning of the text.
   Stopwords are common words like "and", "the", "is", etc., which are often removed to focus on more meaningful words.

## Models

The project employs ensemble methods such as Gradient Boosting, Random Forest, and XGBoost for text classification. Models are trained on different sections of Craigslist posts, including for-sale, housing, services, and community. Each section has its trained models to predict the most relevant category based on the heading provided.

## Project Structure

1. **app.py**: Streamlit application for user interaction.
   This file contains the main code for the Streamlit web application, allowing users to input their data and view the predicted category.

2. **codes/**: Contains Python scripts for text preprocessing and section preprocessing.
   This directory holds the scripts responsible for preprocessing the text data and preparing it for model input.

3. **model/**: Contains trained models for text classification.
   This directory stores the trained machine learning models used for predicting the category of Craigslist posts.

4. **utils/**: Contains utilities such as stemmer and lemmatizer.
   This directory contains helper functions and utilities used in the preprocessing steps, such as stemming and lemmatization.

## Dependencies

- Streamlit: For building interactive web applications.
- NLTK: For natural language processing tasks such as tokenization and stemming.
- scikit-learn: For machine learning algorithms and preprocessing tasks.
- XGBoost: For gradient boosting algorithms.
