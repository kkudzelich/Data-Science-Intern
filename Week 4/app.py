import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model и preprocessor
model = pickle.load(open('best_xgb_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_profit_per_trip():
    data = request.form

    # Extract data from form
    city = data['City']
    payment_mode = data['Payment_Mode']
    gender = data['Gender']
    company = data['Company']
    km_travelled = float(data['KM_Travelled'])
    age = int(data['Age'])
    income = float(data['Income'])
    year = int(data['Year'])
    month = int(data['Month'])
    day = int(data['Day'])

    # Create DataFrame with new data
    new_data = pd.DataFrame({
        'City': [city],
        'Payment_Mode': [payment_mode],
        'Gender': [gender],
        'Company': [company],
        'KM Travelled': [km_travelled],
        'Age': [age],
        'Income (USD/Month)': [income],
        'Year': [year],
        'Month': [month],
        'Day': [day]
    })

    # Transform new data
    new_data_transformed = preprocessor.transform(new_data)

    # Prediction
    prediction = model.predict(new_data_transformed)[0]
    # Convert to float
    prediction = round(float(prediction), 2)

    return render_template('index.html', prediction=prediction, data=data)

if __name__ == '__main__':
    app.run(debug=True)