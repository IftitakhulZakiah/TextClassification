from flask import Flask, render_template, flash, request
from wtforms import Form, SubmitField, SelectField, IntegerField, validators, StringField
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)
app.secret_key = 'development_key'

class predictForm(Form):
    tweet = StringField('Tweet')
    submit = SubmitField('Submit')
    
def tokenize_input(data):
    data = data.replace('\f', ' ')
    data = data.replace('\t', ' ')
    data = data.replace('\n', ' ')
    data = data.replace('\r', ' ')
    stop_words = stopwords.words('english') + list(punctuation)
    words = word_tokenize(data.lower())
    word_tokens = [word for word in words if word not in stop_words] 
    sent_tokens = sent_tokenize(data)
    return word_tokens, sent_tokens 

def importance_sent(word_tokens,sentence_tokens):
    word_dist = Counter(word_tokens)
    importance = defaultdict(int)
    for i in range(len(sentence_tokens)):
        for word in word_tokenize(sentence_tokens[i].lower()):
            if word in word_dist:
                importance[i] += word_dist[word]
    return importance

def summarize(importance, sentences, length):
    idx = nlargest(length, importance, key=importance.get)
    summary = [sentences[i] for i in sorted(idx)]
    final_summary = ' '.join(summary)
    return final_summary 

def classification_tweet(tweet):
	stop_words = stopwords.words("english")
	stemmer = SnowballStemmer('english')
	tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.1, stop_words=stop_words, use_idf=True, tokenizer=tokenize_stem)
	tfidf_matrix = tfidf_vectorizer.fit_transform([[tweet]])
	return tfidf_matrix

@app.route("/index", methods=['GET','POST'])
def index():
    form = predictForm(request.form)
    summary = ''		
    if request.method == 'POST':
    	tweet = form.tweet.data
    	# tfidf_matrix = classification_tweet(tweet)
    	word_tokens, sent_tokens = tokenize_input(tweet)
    	sentence_imp = importance_sent(word_tokens, sent_tokens)
    	summary = summarize(sentence_imp, sent_tokens, 4)
        # with open(app.root_path + '/static/finalized_model.pickle','rb') as handle:
        # 	model = pickle.load(handle)
        # classification = model.predict([[tfidf_matrix]])
    return render_template('index.html', form=form, summary=summary)    
    # return render_template('index.html', form=form, summary=summary, classification=areas[classification[0]])

if __name__ == "__main__":
    app.run('127.0.0.1',debug=True)