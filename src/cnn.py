#!/usr/bin/env python
# coding: utf-8
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_classifier():
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    model = Sequential()
    
    """
    ********** Feature extraction layers ********** 
    """
    
    
    # 1st Convolutional Layer
    model.add(
            Conv2D(filters=96, input_shape=(532,709,4), kernel_size=(11,11),
                     strides=(4,4), padding='valid')
            )
    model.add(Activation('relu'))
        # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())
    
    # 2nd Convolutional Layer
    model.add(
            Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), 
                     padding='valid')
            ) #kernel_size=(5,5) dans le papier
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 3rd Convolutional Layer
    model.add(
            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), 
                     padding='valid')
            )
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 4th Convolutional Layer
    model.add(
            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), 
                     padding='valid')
            )
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(
            Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), 
                     padding='valid')
            )
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    """
    ********** Classification layers **********
    """
    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(124, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 2nd Dense Layer
    model.add(Dense(62))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # 3rd Dense Layer
    model.add(Dense(124))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    
    # Output Layer
    model.add(Dense(10))  
    model.add(Activation('softmax'))
    
    model.summary()
    
    # Compile
    adam=Adam(lr=0.00001, decay=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=['accuracy'])
    
    return model


def process_data(data_path):
    """
    Function that creates the X vector containing the characteristics of the 
    images and the target vector containing the classes, from the path of the 
    folder containing the images
    """
    print('-'*30)
    print('Loading and preprocessing data...')
    print('-'*30)
    classes=os.listdir(data_path)
    imgs=[]
    target=[]
    for classe in classes:
        #path of the folder corresponding to the class
        classe_path=os.path.join(data_path,classe) 
        images=os.listdir(classe_path)
        for k in range(len(images)):
            #to make the images the same size
            img = Image.open(os.path.join(classe_path,images[k])).resize(
                    (709,532)) 
            #we normalize the pixels
            imgs.append(np.array(img)/255)  
            #for each image, we store it's class in the target vector
            target.append(classe) 
    
    nb_imgs=len(imgs)
    s=imgs[0].shape  
    X= np.ndarray((nb_imgs,*s) ,dtype=np.uint8)
    for i in range(nb_imgs):
        X[i]=imgs[i]
        
    #target variable
    le=LabelEncoder()
    le.fit(target)
    #labelencoding
    encoded=le.transform(target)
    #OneHotEncoding
    y=to_categorical(encoded) 
    np.save('classes.npy',le.classes_)
        
    return X, y


def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=20, 
                       save=False, plot=False):
    
    
    print('-'*30)
    print('Training classifier...')
    print('-'*30)
    #save the best model
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', 
                                       save_best_only=True) 
    history=model.fit(X_train, y_train, batch_size=64, epochs=epochs, verbose=1,
                      validation_split=0.2, shuffle=True, 
                      callbacks=[model_checkpoint])
    
    if plot == True:
        #plots         
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train','Test'], loc='upper left')
        plt.show()
    
    if save == True:
        print('-'*30)
        print('Saving weights...')
        print('-'*30)
        model.save('model.h5')
        
    acc=model.evaluate(X_test,y_test)[1]
    print("Accuracy of the model on the test set : {0:.2f}%".format(acc*100))
    
def evaluate(model, X_test, y_test):
    acc=model.evaluate(X_test,y_test)[1]
    print("Accuracy of the model on the test set : {0:.2f}%".format(acc*100))
        

if __name__ == '__main__':
    
    #Training and test on narrowband signals
    classifier = get_classifier()
    X,y = process_data(data_path='framePNG')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    train_and_evaluate(classifier, X_train, y_train, X_test, y_test, 20, True)

    #Test on broadband signals
    X_5000, y_5000 = process_data(data_path='5000msPNG')
    X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_5000, y_5000,
                                                                test_size=0.25)
    evaluate(model=classifier, X_test=X_5000, y_test=y_5000)

    #Training and test on narrow band and broadband data
    print('-'*30)
    print('Training and testing on two databases...')
    print('-'*30)

    X_tot=np.concatenate([X,X_5000], axis=0)
    y_tot=np.vstack([y,y_5000])
    X_train_tot,X_test_tot,y_train_tot,y_test_tot=train_test_split(X_tot, y_tot,
                                                                test_size=0.25)
    train_and_evaluate(classifier, X_train_tot, y_train_tot, X_test_tot, 
                       y_test_tot, 30,True)
    
    #Training on concatenation, test on broadband signals 
    print('-'*30)
    print('Training on two databases, testing ...')
    print('-'*30)

    X_tot2=np.concatenate([X,X_train_5], axis=0)
    y_tot2=np.vstack([y,y_train_5])
    train_and_evaluate(classifier, X_tot2, y_tot2, X_test_5, y_test_5, 30, True)
    
    #Training on broadband signals only
    print('-'*30)
    print('Training and testing on 5000ms database ...')
    print('-'*30)
    train_and_evaluate(classifier, X_train_5, y_train_5, X_test_5, y_test_5, 30,
                       True)