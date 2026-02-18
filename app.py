import numpy as np
from flask import Flask, request, render_template
import joblib
import random

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("model.save")
scaler = joblib.load("transform.save")

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Manual input page
@app.route('/manual')
def manual():
    return render_template('Manual_predict.html')

# Sensor input page
@app.route('/sensor')
def sensor():
    # Generate random sensor values (simulating real sensors)
    sensor_values = {
        'ambient': round(random.uniform(-1, 1), 3),
        'coolant': round(random.uniform(-1.5, 1), 3),
        'u_d': round(random.uniform(-0.5, 1), 3),
        'u_q': round(random.uniform(-1.5, 0.5), 3),
        'motor_speed': round(random.uniform(-1.5, 1), 3),
        'i_d': round(random.uniform(-1, 1.5), 3),
        'i_q': round(random.uniform(-1, 1.5), 3)
    }
    return render_template('Sensor_predict.html', sensor_values=sensor_values)

# Manual prediction
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    # Get values from form
    features = [float(x) for x in request.form.values()]
    X_input = [features]
    
    # Scale and predict
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    
    return render_template('Manual_predict.html', 
                          prediction_text=f'Predicted PM temperature: {pred:.2f}')

# Sensor prediction
@app.route('/predict_sensor', methods=['POST'])
def predict_sensor():
    # Get values from hidden inputs
    features = [float(request.form[f]) for f in ['ambient', 'coolant', 'u_d', 'u_q', 
                                                  'motor_speed', 'i_d', 'i_q']]
    X_input = [features]
    
    # Scale and predict
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    
    # Generate new random sensor values for next reading
    sensor_values = {
        'ambient': round(random.uniform(-1, 1), 3),
        'coolant': round(random.uniform(-1.5, 1), 3),
        'u_d': round(random.uniform(-0.5, 1), 3),
        'u_q': round(random.uniform(-1.5, 0.5), 3),
        'motor_speed': round(random.uniform(-1.5, 1), 3),
        'i_d': round(random.uniform(-1, 1.5), 3),
        'i_q': round(random.uniform(-1, 1.5), 3)
    }
    
    return render_template('Sensor_predict.html', 
                          sensor_values=sensor_values,
                          prediction_text=f'Predicted PM temperature: {pred:.2f}')

if __name__ == '__main__':
    app.run(debug=True)