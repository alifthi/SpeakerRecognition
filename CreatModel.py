# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:09:55 2022

@author: alifa
alifathi8008@gmail.com
"""
import tensorflow.keras.layers as tkl
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


class Model():
    def __init__(self,X,Y):
        self.model = self.BModel()
        self.X = X
        self.Y = Y
        self.XTest,self.XTrain,self.YTest,self.YTrain = self.splitData()
        
    def BModel(self):   
        model = tf.keras.models.Sequential([
        tkl.Conv1D(16,kernel_size=3,padding='same',input_shape=[800,1],name = 'conv1'),
        tkl.Activation('relu',name = 'relu1'),
        tkl.Conv1D(16, kernel_size=3,padding='same',name = 'conv2'),
        tkl.Activation('relu',name = 'relu2'),
        tkl.MaxPool1D(strides = 2,name = 'pool1'),
        
        tkl.Conv1D(32,kernel_size=3,padding='same',name = 'conv3'),
        tkl.Activation('relu',name = 'relu3'),
        tkl.Conv1D(32, kernel_size=3,padding='same',name = 'conv4'),
        tkl.Activation('relu',name='relu4'),
        tkl.MaxPool1D(strides = 2,name = 'pool2'),
        
        tkl.Conv1D(64,kernel_size=3,padding='same',name = 'conv5'),
        tkl.Activation('relu',name = 'relu5'),
        tkl.Conv1D(64,kernel_size=3,padding='same',name = 'conv6'),
        tkl.Activation('relu',name = 'relu6'),
        tkl.Conv1D(64, kernel_size=3,padding='same',name = 'conv7'),
        tkl.Activation('relu',name = 'relu7'),
        tkl.MaxPool1D(strides = 2,name = 'pool3'),
        
        tkl.Conv1D(128,kernel_size=3,padding='same',name = 'conv8'),
        tkl.Activation('relu',name = 'relu8'),
        tkl.Conv1D(128,kernel_size=3,padding='same',name = 'conv9'),
        tkl.Activation('relu',name = 'relu9'),
        tkl.Conv1D(128, kernel_size=3,padding='same',name = 'conv10'),
        tkl.Activation('relu',name = 'relu10'),
        tkl.MaxPool1D(strides = 2,name = 'pool4'),
        
        tkl.Flatten(),
        tkl.Dense(128,activation='relu',name = 'dense1'),
        tkl.Dense(5,activation='softmax',name = 'dense2')])
        
        return model
        
    def CompileModel(self,Optimizer='adam',Loss='categorical_crossentropy',
                    metric = ['accuracy']):
        self.model.summary()
        self.model.compile(loss=Loss,metrics=metric,optimizer=Optimizer)


    def FitModel(self,e=15,b=128):
        hist = self.model.fit(self.XTrain,self.YTrain,epochs=e,batch_size=b,validation_data=(self.XTest,self.YTest))
        return hist

    def SaveModel(self,path='./Model'):
        self.model.save(path + '/Model.h5')

    def splitData(self):
        split = train_test_split(self.X,self.Y,test_size=0.2)
                
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
        split = [(xtest,xtrain),(ytest,ytrain)]
        return split
