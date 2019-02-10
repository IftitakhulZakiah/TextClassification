from flask import Flask, render_template, flash, request
from wtforms import Form, SubmitField, SelectField, IntegerField, validators, StringField
import pandas as pd
import numpy as np
import pickle
import requests
import re
import json
import unicodedata
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from string import punctuation
# from nltk.probability import FreqDist
# from heapq import nlargest
# from collections import defaultdict
# from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
# from gensim.models import Word2Vec
# from gensim.models.doc2vec import TaggedDocument

app = Flask(__name__)
app.secret_key = 'development_key'

with open(app.root_path + '/static/classifier.joblib','rb') as handle:
	model = joblib.load(handle)
classification = model.predict(tweet)

class predictForm(Form):
    tweet = StringField('Tweet')
    submit = SubmitField('Submit')
    
# def tokenize_input(data):
#     data = data.replace('\f', ' ')
#     data = data.replace('\t', ' ')
#     data = data.replace('\n', ' ')
#     data = data.replace('\r', ' ')
#     stop_words = stopwords.words('english') + list(punctuation)
#     words = word_tokenize(data.lower())
#     word_tokens = [word for word in words if word not in stop_words] 
#     sent_tokens = sent_tokenize(data)
#     return word_tokens, sent_tokens 

# def importance_sent(word_tokens,sentence_tokens):
#     word_dist = Counter(word_tokens)
#     importance = defaultdict(int)
#     for i in range(len(sentence_tokens)):
#         for word in word_tokenize(sentence_tokens[i].lower()):
#             if word in word_dist:
#                 importance[i] += word_dist[word]
#     return importance

# def summarize(importance, sentences, length):
#     idx = nlargest(length, importance, key=importance.get)
#     summary = [sentences[i] for i in sorted(idx)]
#     final_summary = ' '.join(summary)
#     return final_summary 

# def classification_tweet(tweet):
# 	stop_words = stopwords.words("english")
# 	stemmer = SnowballStemmer('english')
# 	tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.1, stop_words=stop_words, use_idf=True, tokenizer=tokenize_stem)
# 	tfidf_matrix = tfidf_vectorizer.fit_transform([[tweet]])
# 	return tfidf_matrix

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
	temp = stemWord(removeStopWord(removedPunctutation(removeTag(removeURL(tweet)))))
	# tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.1, stop_words=stop_words, use_idf=True, tokenizer=tokenize_stem)
	# tfidf_matrix = tfidf_vectorizer.fit_transform([[temp]])
	return temp

@app.route("/index", methods=['GET','POST'])
def index():
    form = predictForm(request.form)
    if request.method == 'POST':
    	tweet = form.tweet.data
    	# tfidf_matrix = extract_tweet(tweet)
    	# word_tokens, sent_tokens = tokenize_input(tweet)
    	# sentence_imp = importance_sent(word_tokens, sent_tokens)
    	# summary = summarize(sentence_imp, sent_tokens, 4)
        # with open(app.root_path + '.../model/classifier.joblib','rb') as handle:
        model = joblib.load('.../model/classifier.joblib')
        # classification = model.predict([[tfidf_matrix]])
        classification = model.predict(tweet)

    return render_template('index.html', form=form, classification=classification)    
    # return render_template('index.html', form=form, summary=summary, classification=areas[classification[0]])

if __name__ == "__main__":
    app.run('127.0.0.1',debug=True)