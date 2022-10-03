#Used for Testing various aspects of the Recon, Hyperspectral, and Material Files
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
import numpy as np
import random

from Materials.Materials import Material as m
from Materials.Materials import Material_Lib
from Hyperspectral import Hyperspectral as h
import Hyperspectral as hs
from Recon import Recon as rc

# hs.show_all_files()
hs.compare_all_files()


# file = hs.AV_Crop_7A

# file = hs.files[4]
# image = h(file) #a hyperspectral image object using hyperspectral
# image = image.reduce_bands() #no arguments will reduce to ~300 - 1000nm range
# image.compare(display=True, save=False)

# image.compare(display=True, save=True)

# image.analysis()
# image.display_NDVI(display=True)
# image.graph_single_subcategory(2)

#---------------------------------------------------------------------
# CROP AN IMAGE - ONLY CURRENTLY WORKS FOR EVNI FILES
#---------------------------------------------------------------------
# image1 = image.crop([300, 720, 1700, 2250])
# image1 = image.crop([100, 200, 100, 200])

#---------------------------------------------------------------------
# COMMON MATERIALS
#---------------------------------------------------------------------
# material_brick = m('1MM0008') #a material object using material
# material_lumber_paint_green = m('1MM0015')
# material_concrete = m('1MM0029')
# material_sand = m('3SL0004')
# material_loam = m('3SL0060')
# material_rock = m('4RK0415')

#---------------------------------------------------------------------
# ADDING ANOMALIES
#---------------------------------------------------------------------
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
# image.display_NDVI()

# image.export('AV-Crop-7A')

#---------------------------------------------------------------------

#Anomaly Detection from Scratch

# image = rc()
# image.rx(A_file_path)

#---------------------------------------------------------------------


















