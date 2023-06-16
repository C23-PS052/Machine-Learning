import flask
from flask import Flask, jsonify, request
import os
import io
import numpy as np
from numpy import asarray
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import csv

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)

# Create a Flask app
app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('model_new.h5')

# Define a route for the API
@app.route('/predict', methods=['GET','POST'])
def predict():
    data = pd.read_csv('data_input.csv')
    X, y = split_sequence(data, n_steps=5)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    n_test = 365-292
    X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]
    
    predictions = model.predict(X_test)
    
    predictions = predictions.reshape(predictions.shape[0])
    
    # Return the predictions as a JSON response
    response = {'prediction': predictions}  # Convert predictions to a list if needed
    return jsonify(response)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)

