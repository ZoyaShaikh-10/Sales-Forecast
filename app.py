import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import encoding


model = pickle.load(open('randomforest.pkl', 'rb'))

app = Flask(__name__)

   


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST': 
        weight = float(request.form['Item_Weight'])
        Item_fat = int(request.form['Item_Fat_Content'])
        visibility = float(request.form['Item_Visibility'])
        mrp = int(float(request.form['Item_MRP']))
        type = int(request.form['Item_Type'])
        year = int(request.form['Outlet_Establishment_Year'])
        size = int(request.form['Outlet_Size'])
        location = int(request.form['Outlet_Location_Type'])
        type = int(request.form['Outlet_Type'])

        int_features = [weight, Item_fat, visibility, type,mrp,  year, size, location, type]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text="Sale Forecast for this Product is : {}".format(output))  

if __name__ == "__main__":
    app.run(debug=True)