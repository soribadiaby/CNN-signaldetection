# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:02:51 2019

@author: Soriba
"""
import os
from PIL import Image, ImageChops
import numpy as np
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from process import get_px, get_burst_start


filename = "AIS_1_lb"
image = filename+".png"
filename_txt = filename+".wav.txt"


def distance(a,b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


def centre(square):
    return (square[0]+square[2])/2, (square[1]+square[3])/2


def intersection(v,w):
    """
    Compute the area of the intersection between two rectangles
    """
    
    X= tf.abs(tf.minimum(tf.maximum(v[0],v[2]), tf.maximum(w[0],w[2])) 
    - tf.maximum(tf.minimum(v[0],v[2]), tf.minimum(w[0],w[2])))
    
    Y = tf.abs(tf.minimum(tf.maximum(v[1],v[3]), tf.maximum(w[1],w[3])) 
    - tf.maximum(tf.minimum(v[1],v[3]), tf.minimum(w[1],w[3])))
    return X*Y


def union(v,w):
    return (
            tf.abs(v[2]-v[0])*tf.abs(v[3]-v[1]) 
            + tf.abs(w[2]-w[0])*tf.abs(w[3]-w[1]) - intersection(v,w)
            )
    
#p is the parameter to return if the intersection is null, to penalize well
def uoi(v1,v2, p):
    
    i = intersection(v1,v2) 
    u = union(v1,v2)
    if i==0:
        return p
    return u/i


def iou(v1,v2):
    i = intersection(v1,v2) 
    u = union(v1,v2)
    return i/u


def custom_metric(y_true, y_pred):
        score=0
        alpha=(532/25)*(709/36)
        for i in range(35):
            for j in range(24):
                yt=y_true[i][j]
                yp=y_pred[i][j]
                
                if yt==0:
                    
                    surface = tf.abs(yt[2]-yt[0])*tf.abs(yt[3]-yt[1])
                    if surface == 0:
                        score+=1
                    else:
                        score+=alpha/surface
                else:
                    score+=iou(yt,yp)
                    
        return score/(36*25)
    

def custom_loss(y_true, y_pred):
    alpha = 2
    l=0
    print(tf.Tensor.eval(y_true))
    for i in range(35):
        for j in range(24):
            yt=y_true[i][j]
            yp=y_pred[i][j]
            if yt==0:
                l+=alpha*tf.abs(yt[2]-yt[0])*tf.abs(yt[3]-yt[1])
            else:
                l+=uoi(yt,yp,1000)
    return l
    

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return bbox  # de la forme (x1, y1, x2, y2)


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def process_data(classe):
    print('-'*30)
    print('Loading and preprocessing data...')
    print('-'*30)
    imgs=[]
    target=[]
    path=os.path.join('/','mnt','disque2','IMT','base','5000msPNG')
    classepath=os.path.join(path,classe)
    images=os.listdir(classepath)
    for k in range(1,len(os.listdir(classepath))):            
            #to make all images the same size
            img = Image.open(os.path.join(classepath,images[k])).resize(
                    (709,532)) 
            #we normalize the pixels
            imgs.append(np.array(img)/255)  
            matrix = np.ndarray((25,36,4), dtype=np.float)
            box = trim(img)
            max_x = box[2] - box[0] + 1
            max_y = box[3] - box[1] + 1
            filename=classe + '_' + str(k) + '.wav.txt'
            bandwidths_px, burst_durations_px = get_px(max_x, max_y)
            #starts of the bursts 
            burst_starts = get_burst_start(classe, filename, max_y, box[1])
            burst_len = burst_durations_px[0]
            
            f0_px = box[0] + np.ceil(max_x / 2)
            bw = bandwidths_px[0]
            columns, lines = 36, 25
            columns_orig, lines_orig = 709, 532
            for k in range(len(burst_starts)):
                components = [f0_px - bw / 2, burst_starts[k],f0_px + bw / 2, 
                             burst_starts[k] + burst_len]
                x = int(np.floor(components[0]*columns/columns_orig))
                y = int(np.floor(components[1]*lines/lines_orig))
                matrix[y,x,:] = components
            target.append(matrix)
            
    nb_imgs=len(imgs)
    s=imgs[0].shape  
    X = np.ndarray((nb_imgs,s[0],s[1],s[2]) ,dtype=np.uint8)
    for i in range(nb_imgs):
        X[i]=imgs[i]
    
    y = np.ndarray((nb_imgs,25,36,4) ,dtype=np.uint8)
    for j in range(nb_imgs):
        y[i]=target[i]
        
    return X, y

X,y=process_data('AIS')
print(X.shape, y.shape)
trainedModel=load_model('/mnt/disque2/IMT/base/model.h5')

#Regressor network

for k in range(12):
    trainedModel.pop()

    
regressor = trainedModel

regressor.add(Dense(64, input_shape = (25,36,384), activation='linear'))

regressor.add(Dense(32, activation='linear'))

regressor.add(Dense(4, activation='linear'))

#Transfer



for layer in regressor.layers[:10]:
    layer.trainable=False
for layer in regressor.layers[10:]:
    layer.trainable=True
    
regressor.compile(loss=custom_loss, optimizer=Adam(lr=0.1),
                  metrics=[custom_metric])
regressor.summary()
regressor.fit(X,y, batch_size=8, epochs=5, verbose = 1)
regressor.save('reg.h5')