'''Examples for Analyzing Spectra in Image and Creating Custom Material's'''

from Materials.Materials import Material as m
from Materials.Materials import Material_Lib
from Hyperspectral_1_0 import Hyperspectral as h
from Recon import Recon as rc
import matplotlib.pyplot as plt


'''Analyzing spectra from a pixel in Image'''
# A_file_path = 'AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
# image_og = h(A_file_path)
# image_og.graph_spectra_area([245, 354, 576, 756])
# l1 = [[92,407], [621,236], [450,1564], [284,1641], [743,1608], [766,2348]]
# for i in range(6):
#     image_og.graph_spectra_pixel(l1[i], 'Clouds', False)
# plt.show()

'''Creating a Material from Image Pixel'''
# filename = '0CM0001.txt'
# cloud_info = ['Cloud', 'Custom', 'Cloud', 'Cloud', 'Large','Large_Cloud',
#               'Kevin McKenzie', '.366 - 2.496', 'AVIRIS Dataset', '6-23-22',
#               'Cloud from AVIRIS Data Image', 'Above', 'X', 'Y',
#               'Wavelength (micrometers)', 'Values from Image', '366', '2600',
#               '224', 'none']
# image_og.create_material([265,704], filename, cloud_info)



'''Comparing materials at different points in image'''
# A_file_path = 'AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
# image_og = h(A_file_path)
# image_og.display_RGB()
# image_og.graph_spectra_pixel([590,2780], 'Not Sure', False)
# image_og.graph_spectra_pixel([621,236], 'Cloud', False)
# image_og.graph_spectra_pixel([669,1073], 'Lake?', False)
# image_og.graph_spectra_pixel([337,2971], 'Uh...', False)