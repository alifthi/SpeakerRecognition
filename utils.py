# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:07:07 2022

plot history
load data
apply fourier transform
save Data 

@author: aifthi
alifathi8008@gamil.com
"""
 
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd

class utils():
    def __init__(self):
        self.dirName = ['Benjamin_Netanyau','Jens_Stoltenberg',
        'Julia_Gillard','Magaret_Tarcher','Nelson_Mandela']
        
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
        

    def readWav(self):
        print('***********************************')

        DF = pd.DataFrame()
        wav=[]
        t=0
        for i in self.dirName:
            print('Reading '+i+' folder...')
            for j in range(1500):
                path = './Data/'+i+'/'+str(j)+'.wav'
                audio = tf.io.read_file(path)
                audio, _ = tf.audio.decode_wav(audio,1, 1600)
                wav.append([audio,t])
            DF=DF.append(wav,ignore_index=True)
            t+=1
            wav=[]

        
        
        print('***********************************')
        print('Columns: \n',self.dirName)
        print('***********************************')
        print('Head: \n',DF.head())
        return [DF,self.dirName]
    @staticmethod
    def SaveDF(DF):
        DF.to_csv('./Data/DataFrame.csv')
    @staticmethod
    def FFT(X):
        for i in range(7500):
            audio = X[i]
            audio = tf.squeeze(audio, axis=-1)

            fft = tf.signal.fft(
                tf.cast(tf.complex(real=audio,
                imag=tf.zeros_like(audio)), 
                tf.complex64))

            X[i]=tf.math.abs(fft[ : (audio.shape[0] // 2)])

        return X