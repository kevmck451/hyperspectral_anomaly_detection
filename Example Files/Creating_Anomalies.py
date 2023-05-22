'''Examples for Creating Anomalies'''
#Do not try to run scripts in this folder, copy cody sections to Hyp_MatSyn.py

from Materials.Materials import Material as m
from Materials.Materials import Material_Lib
from Hyperspectral_1_0 import Hyperspectral as h
import matplotlib.pyplot as plt


'''Creating an Anomaly in image'''
# A_file_path = 'AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
# image = h(A_file_path) #a hyperspectral image object using hyperspectral

# material_brick = m('1MM0008') #a material object using material
material_steel = m('1MM0003')
# material_concrete = m('1MM0029')
# material_asphalt = m('1MM0006')
# material_tar = m('1MM0027')
# material_lumber_paint_green = m('1MM0015')
# material_lumber_paint_black = m('1MM0004')
# material_black_tar_paper = m('1MM0012')
# location_1 = [437, 1779] #dictionary of key-value pair of x-y coordinate of middle of anomaly
# location_2 = [470, 1950]
# location_3 = [550, 2350]
# size = 30   #the distance from the center the shape will extend
#
# image.add_anomaly_2(material_lumber_paint_green, location_1, size)
# image.add_anomaly_2(material_brick, location_2, size+10)
# image.add_anomaly_2(material_concrete, location_3, size+30)
# image.add_anomaly_4(material_brick, location_3, size+30)

#
# #tank
# image.add_anomaly_2(material_black_tar_paper, [550, 2350], 20)
# image.add_anomaly_2(material_black_tar_paper, [560, 2350], 20)
# image.add_anomaly_2(material_black_tar_paper, [525, 2350], 5)
# image.add_anomaly_2(material_black_tar_paper, [515, 2350], 5)
# image.add_anomaly_2(material_black_tar_paper, [505, 2350], 5)
#
# image_og.display_image()
# image.display_image()
# image.display_RGB()
# image.display_Veg_Index()

# #image.export()

'''Adding Anomaly and Graphing Results'''
# material_brick = m('1MM0008') #a material object using material
# location_3 = [550, 2350]
# size = 30   #the distance from the center the shape will extend
# image.add_anomaly_4(material_brick, location_3, size+30)
# image.display_RGB()
# image.graph_spectra_pixel(location_3,'Red Brick',True)


'''Adding Custom Created Material back into Image'''
# cloud = m('0CM0001')
# image_og.add_anomaly_1(cloud, [600, 1300], 100, 1)
# image_og.display_image()
# image_og.graph_spectra_pixel([265,704], 'Cloud', False)
# image_og.graph_spectra_pixel([600, 1300], 'Cloud-Anom', False)
# plt.show()


'''Adding Custom Created Material back into Image'''







