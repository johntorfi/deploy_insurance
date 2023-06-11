from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
import os

model_path = os.path.join(os.path.dirname(__file__), 'best_random_forest_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'age': int(request.json['age']),
        'bmi': float(request.json['bmi']),
        'children': int(request.json['children']),
        'sex_female': 1 if request.json['sex'] == 'female' else 0,
        'sex_male': 1 if request.json['sex'] == 'male' else 0,
        'smoker_no': 1 if request.json['smoker'] == 'no' else 0,
        'smoker_yes': 1 if request.json['smoker'] == 'yes' else 0,
        'region_northeast': 1 if request.json['region'] == 'northeast' else 0,
        'region_northwest': 1 if request.json['region'] == 'northwest' else 0,
        'region_southeast': 1 if request.json['region'] == 'southeast' else 0,
        'region_southwest': 1 if request.json['region'] == 'southwest' else 0
    }

    input_data = pd.DataFrame(data, index=[0])
    input_data = input_data.reindex(columns=['age', 'bmi', 'children', 'sex_female', 'sex_male',
                                             'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest',
                                             'region_southeast', 'region_southwest'], fill_value=0)

    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
