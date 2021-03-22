from flask import Flask, request
import numpy as np
from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("lr.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Let's Predict Bankruptcy 
    ---
    parameters:  
      - name: Net worth/Assets
        in: query
        type: number
        required: true
      - name: Persistent EPS in the Last Four Seasons
        in: query
        type: number
        required: true
      - name: Net profit before tax/Paid-in capital
        in: query
        type: number
        required: true
      - name: Borrowing dependency
        in: query
        type: number
        required: true
      - name: Net Income to Stockholder's Equity
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    var1=float(request.args.get("Net worth/Assets"))
    var2=float(request.args.get("Persistent EPS in the Last Four Seasons"))
    var3=float(request.args.get("Net profit before tax/Paid-in capital"))
    var4=float(request.args.get("Borrowing dependency"))
    var5=float(request.args.get("Net Income to Stockholder's Equity"))
    prediction=classifier.predict([[var1,var2,var3,var4,var5]])
    if prediction==1:
      return "The company is going to be bankrupt!!"
    return "The company is safe. :)"


if __name__=='__main__':
    app.run(host='0.0.0.0')