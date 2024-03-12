import joblib
import numpy as np

# Load models
tfidf_vectorizer_forsale = joblib.load('./model/text/forsale/tfidf.joblib')
gb_model_forsale = joblib.load('./model/text/forsale/models/gb.joblib')
rf_model_forsale = joblib.load('./model/text/forsale/models/rf.joblib')
xgb_model_forsale = joblib.load('./model/text/forsale/models/xgb.joblib')

tfidf_vectorizer_community = joblib.load('./model/text/community/tfidf.joblib')
gb_model_community = joblib.load('./model/text/community/models/gb.joblib')
rf_model_community = joblib.load('./model/text/community/models/rf.joblib')
xgb_model_community = joblib.load('./model/text/community/models/xgb.joblib')

tfidf_vectorizer_housing = joblib.load('./model/text/housing/tfidf.joblib')
gb_model_housing = joblib.load('./model/text/housing/models/gb.joblib')
rf_model_housing = joblib.load('./model/text/housing/models/rf.joblib')
xgb_model_housing = joblib.load('./model/text/housing/models/xgb.joblib')

tfidf_vectorizer_services = joblib.load('./model/text/services/tfidf.joblib')
gb_model_services = joblib.load('./model/text/services/models/gb.joblib')
rf_model_services = joblib.load('./model/text/services/models/rf.joblib')
xgb_model_services = joblib.load('./model/text/services/models/xgb.joblib')

def section_preprocessing(section, heading):
    if section == 'for-sale':
        tfidf_vectorizer = tfidf_vectorizer_forsale
        gb_model = gb_model_forsale
        rf_model = rf_model_forsale
        xgb_model = xgb_model_forsale
        f1_scores = [0.88, 0.91, 0.91]
        config = ['appliances', 'cell-phones', 'photography', 'video-games']

    elif section == 'community':
        tfidf_vectorizer = tfidf_vectorizer_community
        gb_model = gb_model_community
        rf_model = rf_model_community
        xgb_model = xgb_model_community
        f1_scores = [0.77, 0.78, 0.76]
        config = ['activities', 'artists', 'childcare', 'general']

    elif section == 'housing':
        tfidf_vectorizer = tfidf_vectorizer_housing
        gb_model = gb_model_housing
        rf_model = rf_model_housing
        xgb_model = xgb_model_housing
        f1_scores = [0.72, 0.74, 0.72]
        config = ['housing', 'shared', 'temporary', 'wanted-housing']

    elif section == 'services':
        tfidf_vectorizer = tfidf_vectorizer_services
        gb_model = gb_model_services
        rf_model = rf_model_services
        xgb_model = xgb_model_services
        f1_scores = [0.83, 0.84, 0.84]
        config = ['automotive', 'household-services', 'real-estate', 'therapeutic']

    heading = tfidf_vectorizer.transform([heading])

    result_gb = gb_model.predict_proba(heading)[0]
    result_rf = rf_model.predict_proba(heading)[0]
    result_xgb = xgb_model.predict_proba(heading)[0]

    weights = np.array(f1_scores) / sum(f1_scores)
    # Normalize
    weights /= sum(weights)

    # Combine prediction with weighted average
    combined_predictions = np.average([result_gb, result_rf, result_xgb], axis=0, weights=weights)

    predicted_category_index = np.argmax(combined_predictions)
    predicted_category = config[predicted_category_index]
    return predicted_category