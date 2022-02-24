# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:09:55 2022

speaker recognition

@author: alifa
alifathi8008@gmail.com
"""
import VoiceReader as V
import keras as ks
import tensorflow.keras.layers as tkl
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

X = V.readWav()
Y = X[0][1]
X = X[0][0]
X = V.FFT(X)

def BModel():   
    model = ks.models.Sequential([
    tkl.Conv1D(16,kernel_size=3,padding='same',input_shape=[800,1]),
    tkl.Activation('relu'),
    tkl.Conv1D(16, kernel_size=3,padding='same' ),
    tkl.Activation('relu'),
    tkl.MaxPool1D(strides = 2),
    
    tkl.Conv1D(32,kernel_size=3,padding='same'),
    tkl.Activation('relu'),
    tkl.Conv1D(32, kernel_size=3,padding='same' ),
    tkl.Activation('relu'),
    tkl.MaxPool1D(strides = 2),
    
    tkl.Conv1D(64,kernel_size=3,padding='same'),
    tkl.Activation('relu'),
    tkl.Conv1D(64,kernel_size=3,padding='same'),
    tkl.Activation('relu'),
    tkl.Conv1D(64, kernel_size=3,padding='same' ),
    tkl.Activation('relu'),
    tkl.MaxPool1D(strides = 2),
    
    tkl.Conv1D(128,kernel_size=3,padding='same'),
    tkl.Activation('relu'),
    tkl.Conv1D(128,kernel_size=3,padding='same'),
    tkl.Activation('relu'),
    tkl.Conv1D(128, kernel_size=3,padding='same' ),
    tkl.Activation('relu'),
    tkl.MaxPool1D(strides = 2),
    
    tkl.Flatten(),
    tkl.Dense(128,activation='relu'),
    tkl.Dense(5,activation='softmax')])
    
    return model
def CompileModel(model,Optimizer='adam',Loss='categorical_crossentropy',
                 metric = ['accuracy']):
    model.summary()
    model.compile(loss=Loss,metrics=metric,optimizer=Optimizer)
    return model

def FitModel(model,X,Y,XT,YT,e=15,b=128):
    hist = model.fit(X,Y,epochs=e,batch_size=b,validation_data=(XT,YT))
    return hist

def SaveModel(model,name='Model'):
    model.save(name)

def plotHist(hist):
    plt.figure()
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('Model accuracy')
    
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('Model loss')


xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
xtrain = xtrain.reset_index(drop=True)
ytrain = ytrain.reset_index(drop=True)
ytrain = tf.keras.utils.to_categorical(ytrain,5)

xtest = xtest.reset_index(drop=True)
ytest = ytest.reset_index(drop=True)
ytest = tf.keras.utils.to_categorical(ytest,5)

for i in range(len(xtrain)):
    xtrain[i] = list(np.asarray(xtrain[i]))

    for t in range(len(xtrain[i])):
        xtrain[i][t] = float(xtrain[i][t])
        
for i in range(len(xtest)):
    xtest[i] = list(np.asarray(xtest[i]))
    for t in range(len(xtest[i])):
        xtest[i][t] = float(xtest[i][t])

    
xtrain = xtrain.tolist()
ytrain = ytrain.tolist()

xtest = xtest.tolist()
ytest = ytest.tolist()

model = BModel()
CompileModel(model)
hist = FitModel(model,xtrain,ytrain,xtest,ytest)
plotHist(hist)
SaveModel(model)
