from flask import Flask, request, jsonify
import joblib
import numpy as np
import threading
import requests

app = Flask(__name__)


model = joblib.load('diabetes_model.pkl')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            
            data = request.get_json()

            if data is None:
                raise ValueError("Dados ausentes na solicitação")

            
            input_data = np.array([[
                data['Pregnancies'],
                data['Glucose'],
                data['BloodPressure'],
                data['SkinThickness'],
                data['Insulin'],
                data['BMI'],
                data['DiabetesPedigreeFunction'],
                data['Age']
            ]])

            prediction = model.predict(input_data)

            return jsonify({
                "prediction": int(prediction[0]),
                "input_features": data  
            })

        elif request.method == 'GET':
            return "Rota '/predict' aceita solicitações GET e POST."

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

