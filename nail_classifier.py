from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import keras
import numpy as np


app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')
test_sample = np.load('test_one_sample.npz')['arr_0']
class NailClassifier(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        
        # vectorize user query and make prediction
        output_vectorized = test_sample
        model = load_model('first_CNN_Performance_best.h5')
        prediction_prob = model.predict(output_vectorized)[0]
        prediction = int(np.where(prediction_prob == np.amax(prediction_prob))[0])
        print(prediction)
        print(prediction_prob)

        # Output either 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_text = 'Bad Nail'
        else:
            pred_text = 'Good Nail'

        # round the predict proba value and set to new variable
        confidence = round(prediction_prob[prediction], 3)
        # create JSON object
        output = {'prediction': pred_text, 'confidence': int(confidence)}
        return output

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(NailClassifier, '/')

if __name__ == '__main__':
    app.run(debug=True)
