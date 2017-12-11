"""basic flask app that returns a gender to a name"""
import nltk
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.externals import joblib
from flask import Flask, request
app = Flask(__name__)

MODEL = joblib.load('boys_girls_names.pkl')
FEATURES = joblib.load('features_list.pkl')
RESULT = 1
NAME = 0

@app.route('/')
def hello():
    return "hello, to get the gender's of a list of names,  " \
           "add /gender/name1,name2,name3... to the end of the url"

@app.route('/gender/<string:names>')
def list_names(names=""):
    """takes a string from a get request that is delimited by a , and returns a list"""
    list = names.split(',')
    return "\n".join(gender(list))


def gender(words):
    """uses model to evaluate the gender of American names"""
    names = []
    grammed_list = []
    responses = []
    for word in words:
        grammed_name = Counter([''.join(ngram) for ngram in nltk.ngrams(str(word), 2)])
        grammed_list.append(grammed_name)
        names.append(word)
    sample = pd.DataFrame(grammed_list, columns=FEATURES).fillna(0).astype(int)
    results = MODEL.predict(sample)
    for value in list(zip(names, results)):
        responses.append("{} is a girl's name".format(value[NAME])) if value[RESULT] == True \
            else responses.append("{} is a boy's name".format(value[NAME]))
    return responses

