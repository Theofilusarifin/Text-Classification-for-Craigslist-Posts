import re
import string
import nltk
from nltk.tokenize import word_tokenize
import emoji
import joblib

def split_camel_case_hashtag(text):
    words = text.split()
    for i, word in enumerate(words):
        if word.startswith('#'):
            # Remove the leading '#' and split camel case words
            words[i] = words[i].replace('#', '')
            words[i] = re.sub(r'([a-z])([A-Z])', r'\1 \2', words[i])
    return ' '.join(words).lower()

def remove_string_emoticons(text):
    # Remove emoticons (e.g., :), :-), :D, ...)
    emoticon_pattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P)'
    text = re.sub(emoticon_pattern, '', text)

    return text


def remove_emoji(text):
    # Mengubah emoji menjadi string
    # 😆 -> :happy:
    text = emoji.demojize(text)

    # Menghapus string emoji
    text = re.sub(r':[a-z_]+:', '', text)
    return text

def clean_URLs(text):
    return re.sub(r"((www.[^s]+)|(http\S+))","",text)


def clean_punctuations(text):
    punctuations = string.punctuation
    translator = str.maketrans(punctuations.replace("'", ''), ' ' * (len(punctuations) - 1))
    text = text.translate(translator)
    text = re.sub(' +', ' ', text) #Remove extra space
    text = re.sub(r'\'', '', text)
    return text

def clean_numeric(text):
    return re.sub('[0-9]+', '', text)

# Initialize the Porter stemmer and WordNet lemmatizer
stemmer = joblib.load('./utils/stemmer.joblib')
lemmatizer = joblib.load('./utils/lemmatizer.joblib')

nltk.data.path.append('./utils/nltk_data/')

# Get English stopwords
stop_words = {"a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"}

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Perform stemming, lemmatization, and stopword removal
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def text_preproccessing(heading):
    heading = split_camel_case_hashtag(heading)
    heading = heading.lower()
    heading = remove_string_emoticons(heading)
    heading = remove_emoji(heading)
    heading = clean_URLs(heading)
    heading = clean_punctuations(heading)
    heading = clean_numeric(heading)
    heading = preprocess_text(heading)
    return heading    