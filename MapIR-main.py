'''Analysis of Data from MapIR Camera'''

import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
from MapIR import MapIR as mir
from MapIR import MapIRChromTest as mct
from Hyperspectral import Hyperspectral as h
import Hyperspectral as hs

# raw_file_path = 'RGN Files/727 Section.png'
# raw_file_path = 'RGN Files/818 Large Area.png'
# raw_file_path = 'RGN Files/727 Section-Cal.png'
# raw_file_path = 'RGN Files/818 Section.png'


# image = mir(raw_file_path)
# print(image.data)
# image.display_image()

# image.NDVI_Mm(max_val=0.50)
# image.NDVI()
# image.SAVI(0.5)
# image.SAVI(0)
# image.EVI_2()
# image.MSAVI()


# location = [300, 300]
# title = ''
#
# for x in range(1000):
#     location[0] = location[0] + x
#     location[1] = location[1] + x
#     image.graph_spectra_pixel((location), title, False)
# plt.show()


#------------------------------------------------
# Test 1
#------------------------------------------------

# raw_file_path = 'RGN Files/MonoChrom Test/650.jpg'
# image = mir(raw_file_path)
# image.display_image()

# middle = (1840, 1480)
# middle = (1990, 1520)
#
# samples = [(1951,1188), (1991,1285), (2000,1415), (2008,1529), (2000,1667),
#             (1943,1813), (1862,1886), (1707,1845), (1626,1707), (1634,1561),
#             (1642,1431), (1659,1285), (1724,1212), (1780,1139), (2089,1099),
#             (2121,1277), (2129,1512), (2113,1707), (2048,1918), (1886,2089),
#             (1683,2040), (1529,1910), (1504,1707), (1488,1464), (1529,1245),
#             (1642,1107), (1764,936), (1951,928)]

# test = mct('RGN Files/MonoChrom Test')
# test.get_values_pixel(middle)
# test.graph()

# test2 = mct('RGN Files/MonoChrom Test')
# test2.get_values_area(samples)
# test2.graph()

#------------------------------------------------
# Test 2
#------------------------------------------------
# raw_file_path = 'RGN Files/MonoChrom Test 2/650.jpg'
# image = mir(raw_file_path)
# image.display_image()

# middle = (1912, 1162)
#
# samples = [(1571,1146), (1618,1138), (1672,1138), (1734,1146), (1788,1154),
#             (1850,1154), (1896,1162), (1935,1169), (2020,1169), (2059,1169),
#             (2113,1169), (2190,1177)]
#
#
# test = mct('RGN Files/MonoChrom Test 2')
# test.get_values_pixel(middle)
# test.graph()
#
# test2 = mct('RGN Files/MonoChrom Test 2')
# test2.get_values_area(samples)
# test2.graph()

#------------------------------------------------
# Test 3
#------------------------------------------------
raw_file_path = 'RGN Files/MonoChrom Test 3/650.jpg'
image = mir(raw_file_path)
image.display_image()

middle = (1912, 1162)

samples = [(1571,1146), (1618,1138), (1672,1138), (1734,1146), (1788,1154),
            (1850,1154), (1896,1162), (1935,1169), (2020,1169), (2059,1169),
            (2113,1169), (2190,1177)]

test3 = mct('RGN Files/MonoChrom Test 3')
test3.get_values_area(samples)
test3.graph()

#------------------------------------------------
# Pika Info
#------------------------------------------------

# file = 'RGN Files/Pika T2/400nm.bil'
# image = h(file) #a hyperspectral image object using hyperspectral
# image.graph_mapir_pika(display=True, save=False)

# hs.mapir_graph_all()









