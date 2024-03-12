import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier

def section_preprocessing(section, heading):
    if section == 'for-sale':
        tfidf_vectorizer = joblib.load('./model/text/forsale/tfidf.joblib')
        gb_model = joblib.load('./model/text/forsale/models/gb.joblib')
        rf_model = joblib.load('./model/text/forsale/models/rf.joblib')
        xgb_model = joblib.load('./model/text/forsale/models/xgb.joblib')
        config = ['appliances', 'cell-phones', 'photography', 'video-games']
    elif section == 'community':
        tfidf_vectorizer = joblib.load('./model/text/community/tfidf.joblib')
        gb_model = joblib.load('./model/text/community/models/gb.joblib')
        rf_model = joblib.load('./model/text/community/models/rf.joblib')
        xgb_model = joblib.load('./model/text/community/models/xgb.joblib')
        config = ['activities', 'artists', 'childcare', 'general']
    elif section == 'housing':
        tfidf_vectorizer = joblib.load('./model/text/housing/tfidf.joblib')
        gb_model = joblib.load('./model/text/housing/models/gb.joblib')
        rf_model = joblib.load('./model/text/housing/models/rf.joblib')
        xgb_model = joblib.load('./model/text/housing/models/xgb.joblib')
        config = ['housing', 'shared', 'temporary', 'wanted-housing']
    elif section == 'services':
        tfidf_vectorizer = joblib.load('./model/text/services/tfidf.joblib')
        gb_model = joblib.load('./model/text/services/models/gb.joblib')
        rf_model = joblib.load('./model/text/services/models/rf.joblib')
        xgb_model = joblib.load('./model/text/services/models/xgb.joblib')
        config = ['automotive', 'household-services', 'real-estate', 'therapeutic']

    heading = tfidf_vectorizer.transform([heading])

    result_gb = gb_model.predict_proba(heading)[0]
    result_rf = rf_model.predict_proba(heading)[0]
    result_xgb = xgb_model.predict_proba(heading)[0]

    end_result = []

    # Majority Voting 
    for gb, rf, xgb in zip(result_gb, result_rf, result_xgb):
        end_result.append(round((gb+rf+xgb)/3, 2))
    
    predicted_category = config[np.argmax(end_result)]
    return predicted_category