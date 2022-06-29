#Used for Testing various aspects of the Recon, Hyperspectral, and Material Files
#Kevin McKenzie 2022
import numpy as np
import random

from Materials.Materials import Material as m
from Materials.Materials import Material_Lib
from Hyperspectral import Hyperspectral as h
from Recon import Recon as rc
import matplotlib.pyplot as plt

'''Using Recon'''

# A_file_path = 'AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
# A_file_path = 'Anomaly Files/AV-Crop2'
# A_file_path = 'Anomaly Files/AV-Cropped-1'
# A_file_path = 'Anomaly Files/AV-Cropped-2'
# A_file_path = 'Anomaly Files/AVIRIS_A1_Test1'
# image = h(A_file_path) #a hyperspectral image object using hyperspectral

# image1 = image.crop([300, 720, 1700, 2250])
# image1 = image.crop([106, 540, 1066, 1431])
# image1 = image.crop([106, 600, 1066, 1500])


# image.display_image()
# image.display_RGB()
# image.display_Veg_Index()

# image1.display_image()
# image1.display_RGB()
# image1.display_Veg_Index()

# image1.export('AV-Crop2')

# recon_1 = rc(A_file_path)
# recon_1.rx_seg()

# material_brick = m('1MM0008') #a material object using material
# material_lumber_paint_green = m('1MM0015')
# material_concrete = m('1MM0029')
# material_sand = m('3SL0004')
# material_loam = m('3SL0060')
# material_rock = m('4RK0415')


# image1.add_anomaly_2(material_brick, [140, 70], 5)
# image1.add_anomaly_2(material_concrete, [164, 237], 3)
# image1.add_anomaly_2(material_lumber_paint_green, [103, 290], 2)


# image1.display_RGB()
# image1.display_Veg_Index()

# image1.graph_spectra_pixel([132, 200], 'Pixel from Image', False)
# image1.graph_spectra_pixel([164, 237], 'Concrete', False)
# image1.graph_spectra_pixel([103, 290], 'Green Painted Lumber', False)
# image1.graph_spectra_pixel([140, 70], 'Brick', False)
# plt.show()

# image1.export('AV-Crop-3Anom-S')

# anom_file_path = 'Anomaly Files/AV-Crop-3Anom-S'
#
# recon_1 = rc(anom_file_path)
# recon_1.rx_seg()

#---------------------------------------------------------------------

# image.add_anomaly_6(material_brick, [140, 70], 12, .3)
# image.add_anomaly_6(material_concrete, [164, 237], 10, .3)
# image.add_anomaly_6(material_lumber_paint_green, [103, 290], 8, .3)

# x = np.linspace(125,700,num = 15, dtype=int)
# y = np.linspace(125,2800,num = 15, dtype=int)
#
# for i in x:
#     for j in y:
#         image.graph_spectra_pixel([i, j], ' ', False)
# plt.show()

# image.display_RGB()
# image.display_Veg_Index()

# image.export('AV-Crop-3Anom-N-60')

# anom_file_path = 'Anomaly Files/AV-Crop-3Anom-N-60'
# recon_1 = rc(anom_file_path)
# recon_1.rx_seg()

#---------------------------------------------------------------------

# image.reduce_bands()


#---------------------------------------------------------------------

# image.display_RGB()
# image.display_Veg_Index()

# image.add_anomaly_5(material_sand, [50, 22], 5, .2)
# image.add_anomaly_5(material_sand, [90, 58], 4, .2)
# image.add_anomaly_5(material_sand, [281, 42], 8, .2)
# image.add_anomaly_5(material_loam, [460, 250], 1, .2)
# image.add_anomaly_5(material_loam, [440, 250], 1, .2)
# image.add_anomaly_5(material_loam, [450, 240], 1, .2)
# image.add_anomaly_5(material_loam, [450, 260], 1, .2)
# image.add_anomaly_5(material_rock, [83, 265], 8, .5)
# image.add_anomaly_6([417, 149], 8, .1)
# image.add_anomaly_6([352, 68], 4, .05)

# for i in range(28):
#     x = random.randint(90,115)
#     y = random.randint(320,340)
#     image.add_anomaly_5(material_sand, [x, y], 1, .3)

# image.graph_spectra_pixel([83, 265], 'Rock', True)

# image.display_RGB()
# image.display_Veg_Index()

# image.export('AV-Crop-8A')

#---------------------------------------------
anom_file_path = 'Anomaly Files/AV-Crop-8A'
image = h(anom_file_path) #a hyperspectral image object using hyperspectral

# image.display_RGB()

recon_1 = rc(anom_file_path)
recon_1.rx_seg()



