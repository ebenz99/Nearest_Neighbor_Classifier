# Nearest_Neighbor_Classifier
Classifies CIFAR images with nearest-neighbor classification algorithm

The program *nnclassifier.py* takes CIFAR images and currently does a really bad job at classifying them. 
To do this, it takes the subject image to classify and compares it to 10000 other CIFAR images.
- "compare" curently entails simply subtracting the subject image from each other image, and taking the label from the compared image with the lowest difference

