import numpy as np
import pickle
from flask import Flask, request, render_template

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the saved scaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Collect input data from form
    Open = float(request.form['Open'])
    High = float(request.form['High'])
    Low = float(request.form['Low'])
    Adj_Close = float(request.form['Adj_Close'])
    Volume = float(request.form['Volume'])
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])

    # Create the feature array
    features = np.array([[Open, High, Low, Adj_Close, Volume, year, month, day]])
    
    # Use the saved scaler to transform the input features
    features = scaler.transform(features)

    # Predict using the loaded model
    prediction = model.predict(features).reshape(1, -1)

    return render_template('index.html', output=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
