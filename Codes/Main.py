# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:07:07 2022


@author: aifthi
alifathi8008@gamil.com
"""
 
import tensorflow as tf
from CreatModel import Model
from utils import utils


model = Model()
utils = utils()
X = utils.readWav()
Y = X[0][1]
X = X[0][0]
X = utils.FFT(X)



model.CompileModel()
hist = model.FitModel(model,e=1,b=128)
utils.plotHist(hist)
model.SaveModel()
