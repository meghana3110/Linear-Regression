import joblib
import os

import numpy as np
from flask import Flask, request, jsonify, render_template

# Create flask app
app = Flask(__name__)
model = clf = joblib.load("clf.pkl")




@app.route("/")
def Home():
    return render_template("website.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)[0]
    return render_template("website.html", prediction_text = "The Price is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)



