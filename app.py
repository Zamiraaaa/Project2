import pandas as pd
import numpy as np
from flask import Flask
from flask import render_template
from flask import request
import pickle


#
# # Create flask app
#
# app = Flask(__name__)
#
# # Load the pickle model
# prediction = pickle.load(open("model.pkl","rb"))
#
# # model = pickle.load(open('model_le_category.pkl', 'rb'))
# # df['Category'] = le.transform(df['Category'])
# # model = pickle.load(open('model_le_segment.pkl', 'rb'))
# # df['Segment'] = model.transform(df['Segment'])
# # model = pickle.load(open('model_le_shipmode.pkl', 'rb'))
# # df['Ship Mode'] = model.transform(df['Ship Mode'])
# # model = pickle.load(open('model_le_state.pkl', 'rb'))
# # df['State'] = model.transform(df['State'])
# @app.route("/predict",methods = ["POST"])
# def predict():
#     json = request.json
#     print(json)
#     query_df = pd.DataFrame(json)
#     #print(query_df)
#     prediction = model.predict(query_df)
#     return jsonify({"Prediction": list(prediction)})
#
# if __name__ ==  "__main__":
#     app.run(debug=True)



import pickle
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return 'Welcome to Profit Pricing Solution API'

@app.route("/predict",methods = ["GET"])
def predict():

    Category = request.args.get('Category')
    Discount = request.args.get('Discount')
    Profit = request.args.get('Profit')
    Quantity = request.args.get('Quantity')
    Sales = request.args.get('Quantity')
    Segment = request.args.get('Segment')
    ShipMode = request.args.get('Ship Mode')
    State = request.args.get('State')

    makeprediction = model.predict([[Category,Discount,Profit,Quantity,Sales,Segment,ShipMode,State]])
    return jsonify({'Profit:': makeprediction})
    new_data = {'X': [6, 7, 8]}
    new_df = pd.DataFrame(new_data)
    predictions = model.predict(new_df[['X']])
    return jsonify(predictiction)
#
#
# if __name__ == '__main__':
#     app.run(debug=True)

# Create a sample DataFrame

from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 22]}
df = pd.DataFrame(data)

@app.route('/get_dataframe', methods=['GET'])
def get_dataframe():
    df_json = df.to_json(orient='records')
    return df_json

@app.route('/post_dataframe', methods=['POST'])
def post_dataframe():
    global df  # Add this line to access the global variable
    createName = request.json["Name"]
    createAge = request.json["Age"]

    new_row = {'Name': createName, 'Age': createAge}
    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)

    return new_df.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
