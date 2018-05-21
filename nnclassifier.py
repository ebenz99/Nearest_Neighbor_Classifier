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
		dif = 10000
		label = "default"
		for x in range(len(self.labs)):
			im = self.ims[x]
			lab = self.labs[x]
			newDif = np.ndarray.sum(np.subtract(im,image))
			if newDif < dif:
				dif = newDif
				label = lab
		return label


#unpacks the CIFAR data
currDir = os.getcwd()
train_data = unpickle(currDir + "\\cifar-10-batches-py\\data_batch_1")
test_data = unpickle(currDir + "\\cifar-10-batches-py\\data_batch_2")
meta = unpickle(currDir + "\\cifar-10-batches-py\\batches.meta")


#turns images from batch 1 into np array of train images
firstPicArr = train_data[b'data']
npPicArr = np.asarray(firstPicArr, dtype=np.float32)
ims = npPicArr.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")


firstCatArr = train_data[b'labels']
npCatArr = np.asarray(firstCatArr)
cats = npCatArr

TrainSet = pipeDream(ims,cats)
testedLab = TrainSet.findClosestMatch(ims[0])

#idx = random.randint(0,10000)
idx = 0
print(ims[idx].shape)
plt.imshow(ims[idx], interpolation="nearest")
plt.title((meta[b'label_names'])[cats[idx]].decode('utf-8'))
plt.show()

plt.imshow(ims[idx], interpolation="nearest")
plt.title((meta[b'label_names'])[testedLab].decode('utf-8'))
plt.show()
