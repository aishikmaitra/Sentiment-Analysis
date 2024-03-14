import numpy as np
from numpy import array

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
import re
from nltk.tokenize import word_tokenize
import nltk
np.random.seed(7)
top_words=5000
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=top_words)
wordtoid=imdb.get_word_index()
Rev_length=500
X_train=sequence.pad_sequences(X_train,maxlen=Rev_length)
X_test=sequence.pad_sequences(X_test,maxlen=Rev_length)
vector_level=32
model=Sequential()
#Embedding layer
model.add(Embedding(top_words,vector_level,input_length=Rev_length))
#Dropout
model.add(Dropout(0.2))
#Model
model.add(LSTM(100))
#Dropout
model.add(Dropout(0.2))
#Dense Layer
model.add(Dense(1,activation='sigmoid'))

#Loss
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30,batch_size=64)

scores=model.evaluate(X_test,y_test,verbose=0)
print("Accuracy:%.2f%%" % (scores[1]*100))

model.save("SenAna2.h5")



