import numpy as np
from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
 
@app.route('/predict', methods=['POST'])
def predict():
    request_body = request.get_json()
    
    fixed_acidity = request_body['fixed acidity']
    volatile_acidity = request_body['volatile acidity']
    chlorides = request_body['chlorides']
    fsd = request_body['free sulfur dioxide']
    ph = request_body['pH']
    alcohol = request_body['alcohol']
    prediction = model.predict([[fixed_acidity, volatile_acidity, chlorides, fsd, ph, alcohol]])[0]
    
    return jsonify({'Wine Score': np.round(prediction)})
