from os import name
import numpy as np
from flask import Flask, request, jsonify, render_template
import urllib.request
from math import expm1
import pandas as pd
import pickle
import nlp_tools
import contractions
from sklearn.feature_extraction.text import CountVectorizer
import logging
from logging.handlers import RotatingFileHandler
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


@app.route('/speechCapture', methods = ['GET','POST'])
def speechCapture():  
    return render_template('speech_capture.html')



@app.route('/predict', methods = ['GET','POST'])
def predict():
    filename = r'feedbackmodel.pkl'
    filename1 = r'count_vectorizer.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    cv = pickle.load(open(filename1, 'rb'))
    
    if request.method == "POST":
        textvalue = request.form.get('note-textarea')
        value1 = "hellow"
        loaded_model = pickle.load(open(filename, 'rb'))
        clean_review = contractions.expand_contraction(textvalue)
        lemma_review = nlp_tools.lemmatization_sentence(clean_review)
        vector_review = cv.transform([lemma_review]).toarray()
        review = loaded_model.predict(vector_review)
        pred = loaded_model.predict_proba(vector_review)
        if review[0] == 0:
            answer = "bad"
        else:
            answer = "good"
    return render_template('speech_capture - Copy.html', value2 = review)


if __name__=='__main__':
   app.run(debug=True)