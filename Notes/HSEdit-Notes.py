import numpy as np
import matplotlib.pyplot as plt

data = open('AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img', 'rb')
    #opens a file and returns it as a file object
    #arguments: files, mode
        #mode: 'r' - Read: default / 'b' binary mode (images)

#print(data)
#<_io.BufferedReader name='/Volumes/KM1TB/HS Data/AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'>

#print(type(data))
#<class '_io.BufferedReader'>

#print(data.readline())
#ff\xce\xff\xce\xff\xce\xff\xce\xff\xce\xff
#binary data held in a Python bytes object
#\x is usually used to represent ascii chars using their hex values in 2 digit pairs

#data = np.frombuffer(data.read(), ">i2")
#print(data)
#[-50 -50 -50 ... -50 -50 -50]
#Integers

data = np.frombuffer(data.read(), ">i2").astype(np.float32)
    #np.frombuffer is convert the bile of bytes to a numbpy array with headers
    #arguments: where the bytes are located (data and we want to read it so data.read),
    #'>i2' is the data type (dtype) > means big-endian and i2 means signed 2-byte integer
    #astype function converts the 2-byte integer into a 32bit float type

#print(data)
#[-50. -50. -50. ... -50. -50. -50.]
#print(type(data))
#<class 'numpy.ndarray'>

data = data.reshape(3077, 869, 224)

#print(data)
#organized it into a 3d array based on the specs above
#The specs for the dimensions can be found in the .hdr file
#print(type(data))

data[1000:2000, 69:800, 10] = 0
#selecting the areas: first is up and down (y), next is side to side (x) and 3rd is the band
#set those pixel values equal to 0 or some other value

plt.imshow(data[:, :, 10]/65536)
#show all the values in data for x and y and then show band 10.
#All the values are divided by 65536 which is largest number for 16 bits
#This turns all the values into numbers less than 1
#
plt.show()







