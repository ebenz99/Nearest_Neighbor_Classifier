import pickle
import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
from PIL import Image
import random

#loads CIFAR images from file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#class to hold train data set and run comparisons from them
class pipeDream:
	def __init__(self, arrayOfImages, arrayOfLabels):
		self.ims = arrayOfImages
		self.labs = arrayOfLabels
	def findClosestMatch(self, image):
		dif = 1000000
		label = 10
		bestIdx = 0
		for x in range(len(self.labs)):
			im = self.ims[x]
			lab = self.labs[x]
			newDif = np.ndarray.sum(np.subtract(im,image))
			if newDif < dif:
				dif = newDif
				label = lab
				bestIdx = x
		return label


'''TRAINING'''
#unpacks the CIFAR data for training set
currDir = os.getcwd()
train_data = unpickle(currDir + "\\cifar-10-batches-py\\data_batch_1")
meta = unpickle(currDir + "\\cifar-10-batches-py\\batches.meta")

#turns images from batch 1 into np array of train images
picArr1 = train_data[b'data']
npPicArr1 = np.asarray(picArr1, dtype=np.float32)
ims1 = npPicArr1.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

#creates parallel array of numeric labels from batch 1
labArr1 = train_data[b'labels']
npLabArr1 = np.asarray(labArr1)
labs1 = npLabArr1

#create object from images and associated labels
TrainSet = pipeDream(ims1,labs1)


''' TESTING '''
#unpickles test data from batch 2
test_data = unpickle(currDir + "\\cifar-10-batches-py\\data_batch_2")

#turns images from batch 2 into np array of train images
picArr2 = test_data[b'data']
npPicArr2 = np.asarray(picArr2, dtype=np.float32)
ims2 = npPicArr2.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

#makes parallel numeric labels from batch 2 categorization
labArr2 = test_data[b'labels']
npLabArr2 = np.asarray(labArr2)
labs2 = npLabArr2

#for procedure
numRight = 0
newLabels = []
numToRead = 300

#runs nearest neighbor classification, compares resulting label to real label
for num in range(0,len(labs2)):
	newLab = TrainSet.findClosestMatch(ims2[num])
	newLabels.append(newLab)
	if newLab == labs2[num]:
		numRight += 1
	if num > numToRead:
		break

#displays accuracy
accuracy = numRight / numToRead
print(accuracy)

