# Use DB as an array for grayscal images
# make train and test sets to array

import os
import numpy as np
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


DB_DIR = '/home/mona/Code/NailClassifier/Data/nailgun/'
DIR = '/home/mona/Code/NailClassifier/ImgArr/'
IMG_H = 250
IMG_W = 250
IMG_CH = 3


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
                        interpolation = 'bicubic')

                ### grayscale to RGB modified for ImageNet.
                new_img = np.stack((img,)*3, axis=-1)
                #imgArray.append(np.reshape(img, [IMG_H, IMG_W, IMG_CH]))
                imgArray.append(new_img)

    imgArray = np.array(imgArray)
    print(imgArray.shape)    
    return imgArray 		
		
if __name__ == '__main__' :
    
    print('Loading images of Test set...')
    dbDir = DB_DIR + 'TestSet/'
    testDBDirs = ('bad/', 'good/')
    
    testLabelSet = []
    for i in range(len(testDBDirs)):
		### Load images into a single array
        testDir = dbDir + testDBDirs[i]
        partialTstImgSet = loadImage(testDir)
        if (i == 0):
	        testImgSet = partialTstImgSet
        else:
            testImgSet = np.concatenate((testImgSet, partialTstImgSet))

		### Make labels of loaded images
        length = len(partialTstImgSet)
        labels = np.ones(length) * i
        testLabelSet.append(labels)
    merged_list = []
    for item in testLabelSet:
        for ele in item:
            merged_list.append(ele)
    testLabelSet = merged_list
    print(testImgSet.shape)


    testImgSet = testImgSet.astype(np.float32)
    testImgSet = (testImgSet-testImgSet.min())/(testImgSet.max()-testImgSet.min())

	# Save images.
    print('Saving images...')
    np.savez(DIR + 'ImgArr_test_try250.npz',
        testImgSet = testImgSet,
        testLabelSet = testLabelSet
	)
 