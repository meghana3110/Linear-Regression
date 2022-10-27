
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template,Response, send_file
import matplotlib.pyplot as plt




# Create flask app
app = Flask(__name__)
model = joblib.load("clf.pkl")
v=pd.read_csv("predvsact.csv")
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
@app.route("/")
def Home():
    return render_template("website.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)[0] 
    return render_template("website.html", prediction_text = "The Price is {}/-Rs".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)