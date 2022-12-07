# Projet 7 : Implémentez un modèle de scoring
import os
import numpy as np
import json
import pickle

from flask import Flask, request

app = Flask(__name__)
app.config["DEBUG"] = True


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
    data = []
    for key in request_json.keys():
        data.append(request_json[key])

    # Loading the model
    model = load_model()

    # Making prediction
        #prediction
    y_proba = model.predict_proba(np.array(X))[0][1]

    # Looking for the customer situation (class 0 or 1)
    # by using the best threshold from precision-recall curve
    y_class = round(y_proba, 2)
    best_threshold = 0.37
    customer_class = np.where(y_class > best_threshold, 1, 0)

    # Customer score calculation
    score = int(y_class * 100)

    # Customer credit application result
    if customer_class == 1:
        result = 'at risk of default'
        status = 'refused'
    else:
        result = 'no risk of default'
        status = 'accepted'

    # API response to the dashboard
    response = json.dumps(
        {'score': score, 'class': result, 'application': status})
    return response, 200


if __name__ == '__main__':
    app.run(debug=True)