import keras
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

fake= pd.read_csv(r"C:\Users\ritur\Downloads\Fake.csv")
true= pd.read_csv(r"C:\Users\ritur\Downloads\True.csv")

fake["label"]= 0
true["label"]= 1

data= pd.concat([fake,true],axis= 0)
data= shuffle(data, random_state=42)

data.reset_index(drop=True,inplace=True)
print(data.head())
print(data.shape)

x= data["text"]
y= data["label"]

vocab_size= 10000
tokenizer= Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x)
x= tokenizer.texts_to_sequences(x)

max_length= 500
x= pad_sequences(x,maxlen= max_length,padding="post")

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = Sequential()
model.add(keras.Input(shape=(max_length,)))
model.add(Embedding(vocab_size,32))
model.add(LSTM(32))
model.add(Dense(1,activation="sigmoid"))
model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test,y_test))
loss, accuracy= model.evaluate(x_test,y_test)
print("Test Accuracy:", accuracy)

model.save("news_model.keras")
file= open("tokenizer.pkl","wb")
pickle.dump(tokenizer,file)
file.close()


