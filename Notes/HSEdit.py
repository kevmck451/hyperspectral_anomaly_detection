#This was the OG File Thomas Edited that one day

import numpy as np
import matplotlib.pyplot as plt

data = open("AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img", "rb")

data = np.frombuffer(data.read(), ">i2").astype(np.float32)

data = data.reshape(3077, 869, 224)

data[1000:2000, 69:800, 10] = 0

plt.imshow(data[:, :, 10])

plt.show()

#--------------------------------------
'''
data = data.astype(">i2")
f = open('AVIRIS-A-L1', "wb")
f.write(data)
'''
#header file needs to be the same name as anomaly file