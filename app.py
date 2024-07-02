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

@app.route("/predict",methods=["POST"])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input= scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The predicted price of House is {} :median value of owner-occupied homes in $1000s.".format(output))

if __name__=="__main__":
    app.run(debug=True)