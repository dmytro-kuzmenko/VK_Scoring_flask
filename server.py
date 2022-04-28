import pandas as pd
from flask import Flask, request, jsonify
from inference import FeatureEncoder, features_to_use
import joblib

app = Flask(__name__)
model = joblib.load('xgb_16f_074-AUC.joblib')


@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # data = test_data

    fe = FeatureEncoder(pd.DataFrame(data, index=[0]), features_to_use)
    fe.encode_features()
    encoded_vector = fe.output_row

    prediction = model.predict_proba(encoded_vector)[:, 0]
    return jsonify(prediction.tolist()[0])


if __name__ == '__main__':
    app.run(port=5000, debug=True)