from Materials.Materials import Material as m
from Materials.Materials import Material_Lib
from Hyperspectral import Hyperspectral as h
from Recon import Recon as rc
import matplotlib.pyplot as plt



'''Using Recon'''


A_file_path = 'AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
# A_file_path = 'Anomaly Files/AV-Cropped-2.hdr'
# A_file_path = 'Anomaly Files/AVIRIS_A1_Test1.hdr'
image = h(A_file_path) #a hyperspectral image object using hyperspectral

# image.display_image()
# image.display_RGB()
image.display_Veg_Index()

# recon_1 = rc(A_file_path)
# recon_1.rx_seg()