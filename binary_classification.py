import numpy as np
import math
from PIL import Image
#MNIST CSV Dataset seems to be 28x28 pixels of handwritten numbers
#First row is from headers, indicating first element in a row is the label or 'name' of the number shown by the pixel data
#The following elements in the header just show the position of each pixel in the 28 by 28 matrix goes from 1x28..28x28

#How about making it into an array of matrices and an array of labels that match indexing or zip them?
#For this I need to get the first column from every row
file = open("./mnist_train.csv", mode="r")
labels = [line.split(",")[0] for line in file.readlines() if line.split(",")[0] != "label"]
print(len(labels))
# with open("./mnist_train.csv", mode="r") as file:
    # print(file.readline())
    

mnist_training_array = np.genfromtxt(fname="mnist_train.csv", delimiter=",", skip_header=1, usecols=range(1, ((28*28)+1)))
# print(mnist_training_array)
for array in mnist_training_array[0:4]:
    reshaped_array = np.reshape(array, (28, 28))
    number_image = Image.fromarray(reshaped_array)
    number_image.show()
    # print(len(array))

#Sigmoid function
for x in range(10):
    sigmoid = 1/(1+math.exp((-1)*x))
    print(sigmoid)
