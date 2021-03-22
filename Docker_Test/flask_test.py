# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 22:39:45 2021

@author: User
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
pickle_in = open("lr.pkl", 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome all"

@app.route('/predict')
def predict_note_authentication():
    var1 = float(request.args.get("var1"))
    var2 = float(request.args.get("var2"))
    var3 = float(request.args.get("var3"))
    var4 = float(request.args.get("var4"))
    var5 = float(request.args.get("var5"))
    
    prediction = classifier.predict([[var1,var2,var3,var4,var5]])
    return "The predicted value is "+ str(prediction)    


if __name__=='__main__':
    app.run(host='0.0.0.0')