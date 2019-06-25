# coding: utf-8
#%env JOBLIB_TEMP_FOLDER=/tmp

from PIL import Image, ImageDraw, ImageChops
import numpy as np
from keras.models import Sequential, load_model, Model
from keras.losses import mse
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, LocallyConnected2D
from keras import backend as K
from keras.optimizers import Adam
from process import *
#from iou import iou
#from iou2 import *
import tensorflow as tf
import os

#os.environ[JOBLIB_SPAWNED_PROCESS]= '1'


#docker run -t -i --env JOBLIB_TEMP_FOLDER=/tmp -v ${pwd}:/data qiime2/core:2019.1 bash


np.set_printoptions(threshold = np.inf)

def mseG(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1,2,3])

def distance(a,b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def centre(carre):
    return (carre[0]+carre[2])/2, (carre[1]+carre[3])/2

    
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
    target=list()
    classes = np.load('classes.npy')
    path=os.path.join('/','mnt','disque2','IMT','base','5000msPNG')
    groundtruthpath=os.path.join('/','mnt','disque2','IMT','base','5000ms',classe)
    classepath=os.path.join(path,classe)
    images=os.listdir(classepath)
    images = sorted(images)
    for k in range(0,len(os.listdir(classepath))):
            img = Image.open(os.path.join(classepath,images[k])).resize((709,532)) #Pour que toutes les images aient la même taille
            imgs.append(np.array(img)/255)  #On normalise les pixels
            matrix = np.zeros((21,32,4), dtype=np.float)
            box = trim(img)
            max_x = box[2] - box[0] + 1
            max_y = box[3] - box[1] + 1
            filename=classe + '_' + str(k+1) + '.wav.txt'
            coordinates=open(os.path.join(groundtruthpath,filename),'r').read().splitlines()
            bandwidths_px, burst_durations_px = get_px(max_x, max_y)
            
            burst_starts = get_burst_start(classe, filename, max_y, box[1])# debuts des bursts pour l'image en question
            burst_len = burst_durations_px[0]
            
            f0_px = box[0] + np.ceil(max_x/2)

            bw = bandwidths_px[0]
            columns, lines= 32, 21
            columns_orig, lines_orig = 709, 532
            for i in range(len(burst_starts)):
                components= [f0_px - bw / 2, burst_starts[i],f0_px + bw / 2, burst_starts[i] + burst_len]
                x,y= int(np.floor(components[0]*columns/columns_orig)), int(np.floor(components[1]*lines/lines_orig))
                deltaX = np.floor(components[0]*columns/columns_orig)
                deltaY = np.floor(components[1] * lines / lines_orig)
                #components = [f0_px - bw / 2-deltaY, burst_starts[i]-deltaX,f0_px + bw / 2-deltaY, burst_starts[i] + burst_len-deltaX]
                matrix[y,x]= components

            target.append(matrix)
            
    nb_imgs=len(imgs)
    s=imgs[0].shape  
    X = np.ndarray((nb_imgs,s[0],s[1],s[2]))
    for i in range(nb_imgs):
        X[i]=imgs[i]
    
    y = np.ndarray((nb_imgs,21,32,4))
    for j in range(nb_imgs):
        y[j]=target[j]
     
    return X, y

classes = ['DMR','EDACS48','EDACS96','NXDN48','NXDN96','ProtocolA','ProtocolB',
 'ProtocolC','ProtocolD']
#We charge the model
trainedModel=load_model('/mnt/disque2/IMT/base/model.h5')

c = 'EDACS48'

print('processing ',c)
X,y=process_data(c)

#####Transfer
ll = trainedModel.layers[9].output
ll = LocallyConnected2D(filters = 32, kernel_size = 5, input_shape=(25, 36,384), activation='linear')(ll)
#ll = Dropout(0.4)(ll)
#ll = BatchNormalization()(ll)
ll = LocallyConnected2D(filters = 16, kernel_size = 1, activation='linear')(ll)
#ll = Dropout(0.4)(ll)
#ll = BatchNormalization()(ll)
ll = LocallyConnected2D(filters = 4, kernel_size = 1, activation='linear')(ll)
model = Model(inputs=trainedModel.input,outputs=ll)

for layer in model.layers[:10]:
    layer.trainable=False
for layer in model.layers[10:]:
    layer.trainable=True
#workers = multiprocessing.cpu_count() - 1
model.compile(loss=mseG, optimizer=Adam(lr=0.01))#,metrics=[competitionMetric2], iou_loss_core, optimizer=Adam(lr=0.005,decay=0.01
model.summary()
model.fit(X,y, batch_size=8, epochs=10, verbose = 1)
model.save_weights('/home/imtg1/Modèles/regressor'+c+'.h5')

#modelSave = Sequential()
#c1 = model.layers[10].output
#c2 = (model.layers[11].output)(c1)
#c3 = (model.layers[12].output)(c2)
#modelSave = Model(inputs=model.layers[10].input,outputs=c3)
#model.save('regressorReduced'+c+'.h5')




#model_json = model.to_json()
#with open("weightsDMR.json",'w') as json_file:
 #   json_file.write(model_json)

#model.save_weights("weightsDMR.h5")

'''

##Test
image = X[0,:,:,:]
grth =  y[0,:,:,:]
image = np.expand_dims(image, axis=0)
bb = model.predict(image, batch_size=1)


##We draw the coordinates of the predicted bounding boxes
path = '/mnt/disque2/IMT/base/5000msPNG/AIS/AIS_01.png'
imgDrw = Image.open(path).resize((709, 532))
box = trim(imgDrw)
max_x = box[2] - box[0] + 1
max_y = box[3] - box[1] + 1
rowsOrig, columnsOrig,dimExtra = np.shape(imgDrw)
draw = ImageDraw.Draw(imgDrw)
print('shape img: ',np.shape(imgDrw))
print('shape bb: ',np.shape(bb))
dim1, rows, columns, coords =np.shape(bb)
rows = 21
columns = 32
deltaX = 0
deltaY = 0
print('shape grth: ',np.shape(grth))
for x in range(rows):
    for y in range(columns):
        draw.rectangle([(bb[0,x,y,0], bb[0,x,y,1]), (bb[0,x,y,2], bb[0,x,y,3])], outline="black")
        draw.rectangle([(grth[x,y,0], grth[x,y,1]), (grth[x,y,2], grth[x,y,3])], outline="blue")
        if (bb[0,x,y,0]>50):
            print([(bb[0,x,y,0], bb[0,x,y,1]), (bb[0,x,y,2], bb[0,x,y,3])])
            print([(grth[x, y, 0], grth[x, y, 1]), (grth[x, y, 2], grth[x, y, 3])])
        deltaY += columnsOrig / columns
    deltaY = 0
    deltaX += rowsOrig/rows

print('img draw ',np.shape(imgDrw))

#print('big values bb',bb[bb[0,:,:,0] > 20])
#We show and save the image
imgDrw.show()
imgDrw.save("drawn_img.png")

'''