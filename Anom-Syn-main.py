#Used for Testing various aspects of the Recon, Hyperspectral, and Material Files
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
import numpy as np
import random

from Materials.Materials import Material as m
from Materials.Materials import Material_Lib
from Hyperspectral import Hyperspectral as h
from Recon import Recon as rc

# A_file_path = 'Datasets/AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
# A_file_path = 'Anomaly Files/AV-Crop'
# A_file_path = 'Anomaly Files/AV-Crop2'
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
# A_file_path = 'Anomaly Files/AV-Crop-9A-68B'

# A_file_path = 'Datasets/Botswana/Botswana.mat'
A_file_path = 'Datasets/Kennedy Space/KSC.mat'
# A_file_path = 'Datasets/Indian Pines/Indian_pines.mat'

# A_file_path = 'Datasets/ABU/abu-airport-2.mat'

# A_file_path = 'Datasets/92AV3C.lan'
# A_file_path = 'Datasets/Pika/d_mosaic.bil'

file_list = ['Datasets/ABU/abu-airport-2.mat',
             'Datasets/ABU/abu-airport-3.mat',
             'Datasets/ABU/abu-airport-4.mat',
             'Datasets/ABU/abu-beach-1.mat',
             'Datasets/ABU/abu-beach-2.mat',
             'Datasets/ABU/abu-beach-3.mat',
             'Datasets/ABU/abu-urban-1.mat',
             'Datasets/ABU/abu-urban-2.mat',
             'Datasets/ABU/abu-urban-3.mat',
             'Datasets/ABU/abu-urban-4.mat',
             'Datasets/ABU/abu-urban-5.mat' ]

# for file in file_list:
#     image = h(file)
    # image.display_RGB()
    # image.display_NDVI()
    # image.display_NDWI()
    # image.display_categories()
    # image.categorize_pixels('S', 20, 50)
    # image.display_image()
    # image.graph_spectra_all_pixels()

image = h(A_file_path) #a hyperspectral image object using hyperspectral
image = image.reduce_bands() #no arguments will reduce to ~300 - 1000nm range
# image.display_RGB()
# image.display_NDVI()
# image.display_NDWI()
# image.display_categories()
# image.display_image()
# image.graph_spectra_all_pixels()
image.categorize_pixels()
image.categorize_pixels('S', 10, 50)



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
























