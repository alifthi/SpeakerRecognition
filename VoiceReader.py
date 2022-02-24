# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:07:07 2022

Read voice and apply fourier transform
@author: aifthi
alifathi8008@gamil.com
"""
  
import tensorflow as tf
import pandas as pd

def readWav():
    print('***********************************')
    DirName = ['Benjamin_Netanyau','Jens_Stoltenberg',
               'Julia_Gillard','Magaret_Tarcher','Nelson_Mandela']
    
    DF = pd.DataFrame()
    wav=[]
    t=0
    for i in DirName:
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
    print('Columns: \n',DirName)
    print('***********************************')
    print('Head: \n',DF.head())
    return [DF,DirName]

def SaveDF(DF):
    DF.to_csv('./Data/DataFrame.csv')
def FFT(X):
    for i in range(7500):
        audio = X[i]
        audio = tf.squeeze(audio, axis=-1)
        fft = tf.signal.fft(
            tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
        )
        X[i]=tf.math.abs(fft[ : (audio.shape[0] // 2)])

    return X