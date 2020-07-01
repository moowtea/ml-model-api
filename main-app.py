import numpy as np
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import pickle

app = Flask(__name__)
api = Api(app)

model = pickle.load(open('model.pkl', 'rb'))

class Predictor(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        
        fixed_acidity = posted_data['fixed acidity']
        volatile_acidity = posted_data['volatile acidity']
        chlorides = posted_data['chlorides']
        fsd = posted_data['free sulfur dioxide']
        ph = posted_data['pH']
        alcohol = posted_data['alcohol']

        prediction = model.predict([[fixed_acidity, volatile_acidity, chlorides, fsd, ph, alcohol]])[0]

        return jsonify({
            'Wine Score': np.round(prediction)
        })
    
api.add_resource(Predictor, '/predict')
