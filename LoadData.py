# Use DB as an array for grayscal images
# make train and test sets to array

import os
import numpy as np
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


DB_DIR = '/home/mona/Code/NailClassifier/Data/nailgun/'
DIR = '/home/mona/Code/NailClassifier/ImgArr/'
IMG_H = 254
IMG_W = 254
IMG_CH = 1


def makeTestList(dbDir):
	if (os.path.isdir(dbDir) == False):
		print('{} does not exist'.format(dbDir))
	else:
		wholeImgList = []
		for _mainDirName, _subDirs, files in os.walk(dbDir):
			for name in files:
				imgFileName = DB_DIR + name
				wholeImgList.append(imgFileName)

			return wholeImgList

def loadImageFromList(imgList, imgArray): 
	for i in range(len(imgList)):
		img = load_img(
					imgList[i],
					grayscale = True,
					target_size = [IMG_H, IMG_W],
					interpolation='bicubic'
				)
		imgArray[i] = np.reshape(img, [IMG_H, IMG_W, IMG_CH])

def loadImage(imgDir): 
    imgArray = []
    if (os.path.isdir(imgDir) == False):
        print('{} does not exist'.format(imgDir))
    else:
        print('Loading images of {}...'.format(imgDir))
        for _mainDirName, _subDirs, files in os.walk(imgDir):
            for name in files:
                imgFileName = imgDir + name
                img = load_img(
						imgFileName,
						grayscale = True,
						target_size = [IMG_H, IMG_W],
						interpolation ='bicubic'
					)
                imgArray.append(np.reshape(img, [IMG_H, IMG_W, IMG_CH]))
				
    imgArray = np.array(imgArray)
    print(imgArray.shape)	
    return imgArray		
		
if __name__ == '__main__' :
    
    print('Loading images of Training set...')
    dbDir = DB_DIR + 'TrainingSet/'
    trainDBDirs = ('bad/', 'good/')

    trainLabelSet = []
    for i in range(len(trainDBDirs)):
		### Load images into a single array
        trainDir = dbDir + trainDBDirs[i]
        partialTrnImgSet = loadImage(trainDir)
        if (i == 0):
	        trainImgSet = partialTrnImgSet
        else:
            trainImgSet = np.concatenate((trainImgSet, partialTrnImgSet))

		### Make labels of loaded images
        length = len(partialTrnImgSet)
        labels = np.ones(length) * i
        trainLabelSet.append(labels)
    merged_list = []
    for item in trainLabelSet:
        for ele in item:
            merged_list.append(ele)
    trainLabelSet = merged_list
    print(trainImgSet.shape)
	
    ### Load images of validationSet
    print('Loading images of validationSet...')
    dbDir = DB_DIR + 'ValidationSet/'
    validationDBDirs = ('bad/', 'good/')

    validationLabelSet = []
    for i in range(len(validationDBDirs)):
        ### Load images into a single array
        validationDir = dbDir + validationDBDirs[i]
        partialValImgSet = loadImage(validationDir)
        if (i == 0):
            validationImgSet = partialValImgSet
        else:
            validationImgSet = np.concatenate((validationImgSet, partialValImgSet))

    ### Make labels of loaded images. 
        length = len(partialValImgSet)
        labels = np.ones(length) * i
        validationLabelSet.append(labels)
    merged_list = []
    for item in validationLabelSet:
        for ele in item:
            merged_list.append(ele)
    validationLabelSet = merged_list
    print(validationImgSet.shape)


    #validationList = makeTestList(validationdbDir)
    #lValidation = len(validationList)
    #validationImgSet = np.ones((lValidation))   

    trainImgSet = trainImgSet.astype(np.float32)
    validationImgSet = validationImgSet.astype(np.float32)
    trainImgSet = (trainImgSet-trainImgSet.min())/(trainImgSet.max()-trainImgSet.min()) 
    validationImgSet = (validationImgSet-validationImgSet.min())/(validationImgSet.max()-validationImgSet.min())

    print('Saving images...')
    np.savez(DIR + 'ImgArrTV.npz',
        trainImgSet = trainImgSet,
        trainLabelSet = trainLabelSet,
        validationImgSet = validationImgSet,
        validationLabelSet = validationLabelSet)
 