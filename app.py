import pickle
from flask import Flask, request, app, render_template, url_for, jsonify
import numpy as np
import pandas as pd

app=Flask(__name__)

#load model
regmodel=pickle.load(open('regression.pkl','rb'))
scalar= pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data= request.json['data']  #whenevet predict.api is hit it will store the data in json format inside data variable
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data= scalar.transform(np.array(list(data.values())).reshape(1,-1)) 
    output= regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)