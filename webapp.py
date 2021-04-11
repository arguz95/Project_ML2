from flask import Flask, request
import numpy as np
from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger
import io
import requests
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split



url = "https://raw.githubusercontent.com/arguz95/Project_ML2/master/Data/bankrupcy.csv"
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8')))
bankruptcy = df.copy()

X = df[['Current Liability to Assets','Debt ratio %','ROA(C) before interest and depreciation before interest',
        'Persistent EPS in the Last Four Seasons','Tax rate (A)']]
y = df['Bankrupt?']



X_train, X_test_final, y_train, y_test_final = train_test_split(X,y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.25, random_state=42)

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
num_attribs = list(X_train)
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs)])

X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)
X_test_final_prepared = full_pipeline.transform(X_test_final)

X_train_prepared = pd.DataFrame(X_train_prepared, columns=num_attribs)
X_test_prepared = pd.DataFrame(X_test_prepared, columns=num_attribs)
X_test_final_prepared = pd.DataFrame(X_test_final_prepared, columns=num_attribs)



app=Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Let's Predict Bankruptcy 
    ---
    parameters:  
      - name: Current Liability to Assets
        in: query
        type: number
        required: true
      - name: Debt ratio %
        in: query
        type: number
        required: true
      - name: ROA(C) before interest and depreciation before interest
        in: query
        type: number
        required: true
      - name: Persistent EPS in the Last Four Seasons
        in: query
        type: number
        required: true
      - name: Tax rate (A)
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    var1=float(request.args.get("Current Liability to Assets"))
    var2=float(request.args.get("Debt ratio %"))
    var3=float(request.args.get("ROA(C) before interest and depreciation before interest"))
    var4=float(request.args.get("Persistent EPS in the Last Four Seasons"))
    var5=float(request.args.get("Tax rate (A)"))
    
    ##Data Pipeline
    values = {'Current Liability to Assets':var1,
              'Debt ratio %':var2,
              'ROA(C) before interest and depreciation before interest':var3,
              'Persistent EPS in the Last Four Seasons':var4,
              'Tax rate (A)':var5}
    
    vals = pd.DataFrame(values,index=[0])
    
    val_new = full_pipeline.transform(vals)
    val_prep = pd.DataFrame(val_new,columns=num_attribs)
    
    
    prediction=classifier.predict(val_prep)
    if prediction>0.5:
      return "The company is going to be bankrupt!!"
    return "The company is safe. :)"


if __name__=='__main__':
    app.run(host='0.0.0.0')
