from os import name
import numpy as np
from flask import Flask, request, jsonify, render_template
import urllib.request
from math import cos, expm1
import pandas as pd
import pickle
import nlp_tools
import contractions
from sklearn.feature_extraction.text import CountVectorizer
import csv
import logging
from logging.handlers import RotatingFileHandler
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/forecasting')
def SalesForecasting():
    return render_template('salesInput.html')

@app.route('/dashboard')
def SalesDashboard():
    return render_template('Salesdashboard.html')


@app.route('/feedbackAnalysis')
def feedbackAnalysis():
    return render_template('speech_capture.html')


@app.route('/feedbackWordCloud')
def feedbackWordCloud():
    return render_template('feedbackwordcloud.html')


@app.route('/speechCapture', methods = ['GET','POST'])
def speechCapture():  
    return render_template('speech_capture.html')



@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    filename = r'feedbackmodel.pkl'
    filename1 = r'count_vectorizer.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    cv = pickle.load(open(filename1, 'rb'))
    length = len(cv.get_feature_names())
    if request.method == "POST":
    
        textvalue = request.form.get('note-textarea1')
        a = "food tasted bad"
        clean_review = contractions.expand_contraction(textvalue)
        lemma_review = nlp_tools.lemmatization_sentence(clean_review)        
        vector_review = cv.transform([lemma_review]).toarray()
        review = loaded_model.predict(vector_review)
        pred = loaded_model.predict_proba(vector_review)
        if review[0] == 0:
            answer = "bad"
        else:
            answer = "good"
        fields = [textvalue, review[0]]
        with open(r'C:\Users\utkar\OneDrive\Desktop\Machine Learning\Customerfeedback\Restaurant_Reviews.tsv','a') as fd:
            writer = csv.writer(fd, delimiter='\t')
            writer.writerow(fields)
    return render_template('speech_capture.html')


@app.route('/salesPrediction', methods = ['POST'])
def salesPrediction():
    store_number = 0
    price = 0
    sold = 0
    unitCost = 0
    cost = 0
    margin = 0
    profit = 0
    predict_values = []
    filename2 = r'salesPrediction.pkl'
    loaded_model = pickle.load(open(filename2, 'rb'))
    if request.method == 'POST':
        loaded_model = pickle.load(open(filename2, 'rb'))
        store_number = request.form.get('storeNumber')
        product = request.form.get('product')
        price = float(request.form.get('Price'))
        sold = request.form.get('Sold')
        unitCost = float(request.form.get('unitCost'))
        cost = float(unitCost) * float(sold)
        margin = float(request.form.get('Margin'))
        profit = float(request.form.get('Profit'))

        predict_values.extend([price, sold, unitCost, cost, margin, profit])
        input_array = np.array(predict_values)
        input_array.reshape(-1,1)
        input_array = input_array.astype(np.float64)
        input_array_for_prediction = np.expand_dims(input_array,axis=0)
        answer = loaded_model.predict(input_array_for_prediction)
    return render_template('salesForecasting.html',storenum = store_number, productsold = sold ,product = product ,value = round(answer[0], 2))


if __name__=='__main__':
   app.run(debug=True)