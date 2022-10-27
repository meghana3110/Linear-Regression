
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template,Response
import json
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO
from matplotlib.figure import Figure
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


x=pd.read_csv("X_values.csv")
y=pd.read_csv("Y_val.csv")

# Create flask app
app = Flask(__name__)
model = joblib.load("clf.pkl")
@app.route("/")
def Home():
    return render_template("website.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)[0] 
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Area vs Price")
    axis.set_xlabel("area")
    axis.set_ylabel("price")
    axis.grid()
    axis.scatter(x['area'].values,y.values)
    axis.scatter(features[0][0],prediction)
    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    

    return render_template("website.html", prediction_text = "The Price is {}/-Rs".format(prediction),image=pngImageB64String)


if __name__ == "__main__":
    app.run(debug=True)