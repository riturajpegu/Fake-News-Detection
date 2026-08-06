

import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.ops.gen_data_flow_ops import padding_fifo_queue

model= load_model("news_model.keras")

file=open("tokenizer.pkl","rb")
tokenizer= pickle.load(file)
file.close()

while True:
    news = input("Enter News = ")
    x=tokenizer.texts_to_sequences([news])
    x=pad_sequences(x,maxlen=500,padding="post")
    prediction=model.predict(x)

    if prediction[0][0]>=0.5:
        print("Prediction: TRUE NEWS")
    else:
        print("Prediction: FALSE NEWS")
    choice= input("DO YOU WANT TO CHECK ANOTHER NEWS? (Y/N):")
    if choice =="N":
        print("HAVE A NICE DAY MATE")
        break

