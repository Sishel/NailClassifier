import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Activation, InputLayer, Flatten
from keras.utils.np_utils import to_categorical
from keras import Model
from keras import layers
from keras import optimizers
import os
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
from keras import applications
import random

epochs = 50
batch_size = 5
seed = 130
np.random.seed(seed) # for reproducibility

DB_DIR = '/home/mona/Code/NailClassifier/'
DIR = '/home/mona/Code/NailClassifier/ImgArr/'

IMG_H = 500    
IMG_W = 500
IMG_CH = 3


def build_model():
    model = Sequential()

    model.add(layers.Conv2D(16, (3, 3), input_shape = (IMG_H, IMG_W, IMG_CH)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size = 2))

    model.add(layers.Conv2D(32, (3,3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size = 2))

    model.add(layers.Conv2D(64, (3,3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size = 2))

    model.add(layers.Flatten())

    model.add(layers.Dense(128))
    model.add(layers.Activation('relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(2))
    model.add(layers.Activation('sigmoid'))

    model.compile(loss = 'binary_crossentropy',
                optimizer = optimizers.rmsprop(),
                #optimizer = optimizers.Adadelta(lr = 1e-4),
                #optimizer = optimizers.Adam(lr = 1e-4),
                #optimizer = optimizers.SGD(lr = 0.000001),
                metrics = ['accuracy'])

    return model


def displayPlot():

    plt.figure(1, figsize = (15,8)) 
    
    plt.subplot(221)  
    plt.plot(history.history['acc'])  
    plt.plot(history.history['val_acc'])  
    plt.title('Model accuracy')  
    plt.ylabel('Accuracy')  
    plt.xlabel('Epoch')  
    plt.legend(['Train Accuracy', 'Validation Accuracy']) 
    plt.suptitle('CNN Performance', fontsize=12)
    
    plt.subplot(222)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('Model loss')  
    plt.ylabel('Loss')  
    plt.xlabel('Epoch')  
    plt.legend(['Train Loss', 'Validation Loss']) 

    plt.show() 
    

if __name__ == "__main__":

    loadFile = DIR + 'ImgArr_test_try2.npz'
    X = np.load(loadFile)
    #trainImgSet = X['trainImgSet']
    #trainLabelSet = X['trainLabelSet']
    #validationImgSet = X['validationImgSet']
    #validationLabelSet = X['validationLabelSet']
    testImgSet = X['testImgSet']
    testLabelSet = X['testLabelSet']
    testLabelSet = to_categorical(testLabelSet)

    #trainLabelSet = to_categorical(trainLabelSet)
    #validationLabelSet = to_categorical(validationLabelSet)

    #print('Shape of trainImgSet: {}'.format(trainImgSet.shape))
    #print('Shape of trainLabelSet: {}'.format(trainLabelSet))

    '''model = build_model()
    model.summary()
    history = model.fit(trainImgSet, trainLabelSet,
                        validation_data = (trainImgSet, trainLabelSet),
                        batch_size = batch_size,
                        epochs = epochs)'''

    model = load_model('first_CNN_Performance_best.h5')
    import ipdb; ipdb.set_trace()
    loss, accuracy = model.evaluate(testImgSet, testLabelSet)
    print(loss)
    print(accuracy)
    #(validationImgSet, validationLabelSet)
    predictedImgSet = model.predict(testImgSet)
    print(predictedImgSet)
    plt.savefig(DB_DIR + 'model_plot.jpg')
    #displayPlot()

    ### create directory to save model
    '''storageDir = DB_DIR + 'first_CNN_Performance.h5'
    if (os.path.isdir(storageDir) == False):
        os.mkdir(storageDir)'''

    #model.save('first_CNN_Performance.h5')