from flask import Flask, render_template, flash, request
from wtforms import Form, SubmitField, SelectField, IntegerField, validators, StringField
import pandas as pd
import numpy as np
import pickle
# from nltk.tokenize import word_tokenize
# import requests
import re
# import json
import unicodedata
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.externals import joblib

app = Flask(__name__)
app.secret_key = 'development_key'

with open(app.root_path + '/static/classifier.joblib','rb') as handle:
	model = joblib.load(handle)
# with open(app.root_path + '/static/count_vectorizer.joblib','rb') as handle:
# 	extractor = joblib.load(handle)

class predictForm(Form):
    tweet = StringField('Tweet')
    submit = SubmitField('Submit')

def removeURL(raw):
    cleanr = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    cleantext = re.sub(cleanr, '', raw)
    return cleantext

def removeTag(raw):
    cleanr = re.compile("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)")
    cleantext = re.sub(cleanr, '', raw)
    return cleantext

def removedPunctutation(text):
    removed = set(string.punctuation)
    return ''.join(w for w in text if w not in removed)

def removeStopWord(text):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    return stopword.remove(text)

def stemWord(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

def extract_tweet(tweet):
	temp = stemWord(removeStopWord(removeTag(removeURL(tweet.encode('utf-8')))))
	extractor = CountVectorizer()
	extractor.fit(temp)
	extracted = extractor.transform([[temp]])
	return extracted

@app.route("/index", methods=['GET','POST'])
def index():
    form = predictForm(request.form)
    result = ''
    if request.method == 'POST':
    	tweet = form.tweet.data
        classification = model.predict(extract_tweet(tweet))
        if classification == 0:
        	result = 'Keluhan'
        elif classification == 1:
        	result = 'Respon'
        else:
        	result = 'Bukan Keluhan/Respon'

    return render_template('index.html', form=form, classification=result)    

if __name__ == "__main__":
    app.run('127.0.0.1',debug=True)