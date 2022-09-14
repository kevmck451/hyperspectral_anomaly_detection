'''Analysis of Data from MapIR Camera'''

import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
from MapIR import MapIR as mir
from MapIR import MapIRChromTest as mct


raw_file_path = 'RGN Files/727 Section.png'
# raw_file_path = 'RGN Files/727 Section-Cal.png'
# raw_file_path = 'RGN Files/818 Section.png'
# raw_file_path = 'RGN Files/MonoChrom Test/650.jpg'

image = mir(raw_file_path)
# print(image.data)
# image.display_image()

# image.NDVI()
# image.SAVI(0.5)
# image.SAVI(0)
# image.EVI_2()
image.MSAVI()

'''
location = [300, 300]
title = ''

for x in range(1000):
    location[0] = location[0] + x
    location[1] = location[1] + x
    image.graph_spectra_pixel((location), title, False)
plt.show()
'''

# middle = (1840, 1480)

middle = (1990, 1520)

samples = [(1951,1188), (1991,1285), (2000,1415), (2008,1529), (2000,1667),
            (1943,1813), (1862,1886), (1707,1845), (1626,1707), (1634,1561),
            (1642,1431), (1659,1285), (1724,1212), (1780,1139), (2089,1099),
            (2121,1277), (2129,1512), (2113,1707), (2048,1918), (1886,2089),
            (1683,2040), (1529,1910), (1504,1707), (1488,1464), (1529,1245),
            (1642,1107), (1764,936), (1951,928)]

# test = mct()
# test.get_values_pixel(middle)
# test.get_values_area(samples)
# test.graph()





