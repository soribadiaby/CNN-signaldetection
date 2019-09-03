import os
from PIL import Image, ImageChops
import numpy as np
from keras.models import load_model, Model
from keras.layers import LocallyConnected2D
from keras import backend as K
from keras.optimizers import Adam
from process import get_px, get_burst_start


def mseG(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1,2,3])


def distance(a,b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


def centre(square):
    return (square[0]+square[2])/2, (square[1]+square[3])/2

    
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        # in the form (x1, y1, x2, y2)
        return bbox  


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
    images = sorted(images)
    for k in range(0,len(os.listdir(classepath))):
            #to make all images the same size
            img = Image.open(os.path.join(classepath,images[k])).resize(
                    (709,532))
            #we normalize the pixels
            imgs.append(np.array(img)/255)  
            matrix = np.zeros((21,32,4), dtype=np.float)
            box = trim(img)
            max_x = box[2] - box[0] + 1
            max_y = box[3] - box[1] + 1
            filename=classe + '_' + str(k+1) + '.wav.txt'
            bandwidths_px, burst_durations_px = get_px(max_x, max_y)
            #bursts starts for the image
            burst_starts = get_burst_start(classe, filename, max_y, box[1])
            burst_len = burst_durations_px[0]
            
            f0_px = box[0] + np.ceil(max_x/2)

            bw = bandwidths_px[0]
            columns, lines= 32, 21
            columns_orig, lines_orig = 709, 532
            for i in range(len(burst_starts)):
                components= [f0_px - bw / 2, burst_starts[i],f0_px + bw / 2, 
                             burst_starts[i] + burst_len]
                x=int(np.floor(components[0]*columns/columns_orig))
                y=int(np.floor(components[1]*lines/lines_orig))
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

#Transfer
ll = trainedModel.layers[9].output
ll = LocallyConnected2D(filters = 32, kernel_size = 5, input_shape=(25, 36,384),
                        activation='linear')(ll)
ll = LocallyConnected2D(filters = 16, kernel_size = 1, activation='linear')(ll)
ll = LocallyConnected2D(filters = 4, kernel_size = 1, activation='linear')(ll)
model = Model(inputs=trainedModel.input,outputs=ll)

for layer in model.layers[:10]:
    layer.trainable=False
for layer in model.layers[10:]:
    layer.trainable=True

model.compile(loss=mseG, optimizer=Adam(lr=0.01))
model.summary()
model.fit(X,y, batch_size=8, epochs=10, verbose = 1)
model.save_weights('/home/imtg1/Modèles/regressor'+c+'.h5')

#modelSave = Sequential()
#c1 = model.layers[10].output
#c2 = (model.layers[11].output)(c1)
#c3 = (model.layers[12].output)(c2)
#modelSave = Model(inputs=model.layers[10].input,outputs=c3)
#model.save('regressorReduced'+c+'.h5')