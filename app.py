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

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from os import path
from PIL import Image
import matplotlib.pyplot as plt

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

@app.route('/salesPrediction', methods = ['POST'])
def wordCloud():
    df = pd.read_csv(r"file path will coe of the final written file", delimiter = "\t")
    text = " ".join(review for review in df.Review)

    # Deciding on the stop words
    str_list = text.split()
    unique_words = set(str_list)
    wrds = list() #An empty list wrds which will append all the STOPWORDS
    for words in unique_words :
            #print('Frequency of ', words , 'is :', str_list.count(words))
            if(str_list.count(words) > 20):
                wrds.append(words)
    #print(wrds)
    # Creating stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(wrds)
    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=40).generate(text)

    # Display the generated image:
    # the matplotlib way:
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(12,8))
    review_result = df.groupby("Liked")         # Group the Likes and Dislikes - Liked Column
    review_result.size().sort_values(ascending=False).plot.bar()
    plt.xticks(rotation=0)
    plt.xlabel("Likes - 1, Dislikes - 0")
    plt.ylabel("Number of Likes / Dislikes")
    plt.show()
    
    return render_template('feedbackwordcloud.html')

if __name__=='__main__':
   app.run(debug=True)