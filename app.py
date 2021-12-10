from os import name
import numpy as np
from flask import Flask, request, jsonify, render_template
import urllib.request
from math import expm1
import pandas as pd
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/forecasting')
def SalesForecasting():
    return render_template('salesForecasting.html')

@app.route('/dashboard')
def SalesDashboard():
    return render_template('Salesdashboard.html')


@app.route('/feedbackAnalysis')
def feedbackAnalysis():
    return render_template('feedbackanalysis.html')


@app.route('/feedbackWordCloud')
def feedbackWordCloud():
    return render_template('feedbackwordcloud.html')


@app.route('/speechCapture')
def speechCapture():
    return render_template('speech_capture.html')


if __name__=='__main__':
   app.run(debug=True)