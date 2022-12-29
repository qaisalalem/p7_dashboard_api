# Projet 7 : Implémentez un modèle de scoring
import os
import numpy as np
import json
import pickle
import shap
from flask import Flask, request, jsonify, Response
import pandas as pd
import plotly.graph_objs as go


app = Flask(__name__)
app.config["DEBUG"] = True

relevant_features= ['POS_SK_DPD_DEF','BUR_DAYS_CREDIT_ENDDATE','BUR_AMT_CREDIT_SUM','BUR_AMT_CREDIT_SUM_DEBT',
                        'BUR_AMT_CREDIT_SUM_OVERDUE','BUR_DAYS_CREDIT_UPDATE','PAY_HIST_NUM_INSTALMENT_VERSION',
                        'PAY_HIST_NUM_INSTALMENT_NUMBER','PAY_HIST_DAYS_INSTALMENT','PAY_HIST_AMT_INSTALMENT',
                        'POS_CNT_INSTALMENT','POS_SK_DPD','NAME_EDUCATION_TYPE_Secondary / secondary special',
                        'PREV_APPLICATION_NUMBER','PREV_AMT_ANNUITY','PREV_AMT_DOWN_PAYMENT',
                        'PREV_AMT_CREDIT','PREV_RATE_DOWN_PAYMENT','PREV_CNT_PAYMENT',
                        'NAME_CONTRACT_TYPE','FLAG_OWN_CAR','NAME_EDUCATION_TYPE_Higher education',
                        'REG_CITY_NOT_LIVE_CITY','CODE_GENDER_F','NAME_FAMILY_STATUS_Married',
                        'BUR_DAYS_CREDIT','BUR_CNT_CREDIT_PROLONG','REGION_RATING_CLIENT',
                        'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH',
                        'PAYMENT_RATE','HOUR_APPR_PROCESS_START','EXT_SOURCE_2',
                        'EXT_SOURCE_3','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE',
                        'DAYS_LAST_PHONE_CHANGE','AMT_ANNUITY','AMT_CREDIT',
                        'AMT_INCOME_TOTAL','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR',
                        'ANNUITY_INCOME_RATE','INCOME_CREDIT_RATE','DAYS_BIRTH',
                        'REGION_POPULATION_RELATIVE']

def load_model():
    """
    This function loads a serialized machine learning file.
    """

    folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(folder, 'model/model.pkl')
    with open(file_path, 'rb') as f:
       model = pickle.load(f)
       model = model['model']
    return model


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    This function is used for making prediction.
    """

    # Input data from dashboard request
    request_json = request.get_json()
    print(request_json)

    # Convert data to a list
    data = [request_json[feature] for feature in relevant_features]

    # Convert data to a DataFrame
    data_df = pd.DataFrame([data], columns=relevant_features)

    # Loading the model
    model = load_model()

    # Calculate the feature importances or SHAP values
    # Feature importance
    model.predict(data_df)
    features_importance = model.feature_importances_
    sorted = np.argsort(features_importance)
    dataviz = pd.DataFrame(columns=['feature', 'importance'])
    dataviz['feature'] = np.array(data_df.columns)[sorted]
    dataviz['importance'] = features_importance[sorted]
    dataviz = dataviz[dataviz['importance'] > 200]
    dataviz.reset_index(inplace=True, drop=True)
    dataviz_dict = dataviz.to_dict(orient='records')

    # Calculate the SHAP values
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(data_df)
    # Create a DataFrame from the Shap values
    shap_df = pd.DataFrame(
        list(zip(data_df[relevant_features].columns, np.abs(shap_values[0]).mean(0))),
        columns=['feature', 'importance'])
    #shap_df = shap_df.sort_values(by='importance', ascending=False)
    shap_df.reset_index(inplace=True, drop=True)
    shap_df = pd.DataFrame(shap_values[0], columns=data_df.columns)
    shap_dict = shap_df.to_dict(orient='records')
    


    # Making prediction
    y_proba = model.predict_proba(([data]))[0][1]
    # Looking for the customer situation (class 0 or 1)
    # by using the best threshold from precision-recall curve
    y_class = round(y_proba, 2)
    best_threshold = 0.37
    customer_class = np.where(y_class > best_threshold, 1, 0)

    # Customer score calculation
    score = int(y_class * 100)

    # Customer credit application result
    if customer_class == 1:
        result = 'At risk of default'
        status = 'Loan is refused'
    else:
        result = 'No risk of default'
        status = 'Loan is accepted'

    # API response to the dashboard
    response = json.dumps(
        {'score': score, 'class': result, 'application': status,
        'dataviz': dataviz_dict, 'shap_features': shap_dict})
    return response, 200


if __name__ == '__main__':
    app.run(debug=True)
