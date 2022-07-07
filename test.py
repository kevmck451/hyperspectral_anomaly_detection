#Used for Testing various aspects of the Recon, Hyperspectral, and Material Files
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
import numpy as np
import random

from Materials.Materials import Material as m
from Materials.Materials import Material_Lib
from Hyperspectral import Hyperspectral as h
from Recon import Recon as rc

# A_file_path = 'AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
# A_file_path = 'Anomaly Files/AV-Crop'
# A_file_path = 'Anomaly Files/AV-Crop2'
# A_file_path = 'Anomaly Files/AV-Cropped-1'
# A_file_path = 'Anomaly Files/Test_Image'
# A_file_path = 'Anomaly Files/Test_Image-2'

# A_file_path = 'Anomaly Files/AV-Crop-3Anom-L'
# A_file_path = 'Anomaly Files/AV-Crop-3Anom-M'
# A_file_path = 'Anomaly Files/AV-Crop-3Anom-S'
# A_file_path = 'Anomaly Files/AV-Crop-7A'
# A_file_path = 'Anomaly Files/AV-Crop-8A'
# A_file_path = 'Anomaly Files/AV-Crop-9A'

# A_file_path = 'Anomaly Files/AV-Crop-3A-L-68B'
# A_file_path = 'Anomaly Files/AV-Crop-3A-M-68B'
# A_file_path = 'Anomaly Files/AV-Crop-3A-S-68B'
# A_file_path = 'Anomaly Files/AV-Crop-7A-68B'
# A_file_path = 'Anomaly Files/AV-Crop-8A-68B'
A_file_path = 'Anomaly Files/AV-Crop-9A-68B'

image = h(A_file_path) #a hyperspectral image object using hyperspectral

# image.display_RGB()

# image1 = image.crop([300, 720, 1700, 2250])
# image1 = image.reduce_bands(1000, 300)
# image1.display_RGB()
# image1.export('AV-Crop-9A-68B')

# recon_1 = rc()
# recon_1.rx_seg(A_file_path)


#---------------------------------------------------------------------

# material_brick = m('1MM0008') #a material object using material
# material_lumber_paint_green = m('1MM0015')
# material_concrete = m('1MM0029')
# material_sand = m('3SL0004')
# material_loam = m('3SL0060')
# material_rock = m('4RK0415')

# image.add_anomaly_6([103, 279], 5, .03)
# image.add_anomaly_6([394, 234], 4, .02)
# image.add_anomaly_6([380, 99], 3, .01)
#
# image.add_anomaly_8(material_loam, [301, 221], 6, .2, 0)
# image.add_anomaly_8(material_concrete, [107, 68], 12, .2, .1)
# image.add_anomaly_8(material_brick, [257, 123], 3, .2, .2)

# for i in range(28):
#     x = random.randint(70,82)
#     y = random.randint(328,375)
#     image.add_anomaly_8(material_sand, [x, y], 1, .2, 0)

# x = np.linspace(0,400,num = 15, dtype=int)
# y = np.linspace(0,400,num = 15, dtype=int)
#
# for i in range(len(x)):
#     for j in range(len(y)):
#         image.graph_spectra_pixel([x, y], ' ', False) #outer edge
# plt.show()

# image.display_RGB()
# image.display_Veg_Index()

# image.export('AV-Crop-7A')

#---------------------------------------------------------------------

#Anomaly Detection from Scratch



























