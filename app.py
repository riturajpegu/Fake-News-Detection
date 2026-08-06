from flask import Flask,render_template,request
import pickle
import keras
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app= Flask(__name__)
model = load_model("news_model.keras")
file = open("tokenizer.pkl","rb")
tokenizer = pickle.load(file)
file.close()

max_length = 500

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    news = request.form["news"]

    x = tokenizer.texts_to_sequences([news])
    x = pad_sequences(x,maxlen=max_length, padding="post")
    prediction = model.predict(x)

    if prediction[0][0] >=0.5:
        result = "TRUE NEWS"
    else:
        result = "FALSE NEWS"
    return render_template("index.html",
    prediction=result,news=news)

if __name__ == "__main__":
    app.run(debug=True)