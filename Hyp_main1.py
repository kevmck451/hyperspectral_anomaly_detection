#Used for Testing various aspects of the Recon, Hyperspectral, and Material Files
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
import numpy as np
import random
import os

from Materials.Materials import Material as m
from Materials.Materials import Material_Lib
from Hyperspectral_1_1 import Hyperspectral as h
import Hyperspectral_1_1 as hs
from Hyperspectral_1_1 import time_class as t
from Hyperspectral_1_1 import record_class as r

process = t('Main')

# record_main = r().compile_detectors()
# hs.show_all_files()
# hs.compare_all_files(hs.az_files)

# file = hs.files[10]
# file = hs.avc_set[6]
# file = hs.AV_Crop_8A
# file = hs.az_files[2]
file = hs.AC_S22

# image = h(file, stats=False) #a hyperspectral image object using hyperspectral
# image.display_image()
# image.display_RGB(display=True)

#---------------------------------------------------------------------
''' PRE-PROCESSING OF UAV IMAGES '''
#---------------------------------------------------------------------
# x1, x2, y1, y2 = 20, 100, 20, 100
# bounds = [x1, x2, y1, y2]
# image = image.crop(bounds)
# image.graph_spectra_all_pixels()
# image.preprocess(bands=(410, 900))
#---------------------------------------------------------------------
''' ANALYSIS OF UAV IMAGES '''
#---------------------------------------------------------------------
# image.display_RGB(display=True)
# image.display_NDVI(display=True, save=False)
# image.display_subcategories(display=False, save=True)

# image.sc_filtered(display=True, band='M', sc_var=32, cutoff_percent=(40, 100))
# image.Anomaly_IF(display=True)
# image.Anomaly_RXSeg(display=False)
# image.Anomaly_RXGlobal(display=False)
# image.Anomaly_LOF_fp(display=True, overlay=False, save=False, ncomps=14, neighbor=1000)
# image.Anomaly_PCA(display=True, save=False, stats=False)
# image.Anomaly_Autoencoder(dp=True, sv=False, tv=90)

# image.overview(display=True, save=False)
# image.Anomaly_Stack_1(display=False, overlay=True, save=False)
# image.Anomaly_Stack_1b(display=False, overlay=True, save=False)
# image.Anomaly_Stack_2(display=False, overlay=False, save=False)
# image.Anomaly_Stack_3(display=True, overlay=True, save=False)

i = 73
for file in hs.AC_S22:
    if i < 89:
        i += 1
        continue
    print(f'------------------ Cube {i} ------------------')
    # file = hs.az_files[file]
    image = h(file, stats=False)
    image.preprocess(bands=(410, 900))
    image.overview(display=False, save=True)
    # image.Anomaly_Stack_3(display=False, overlay=True, save=True)
    i+=1

os.system("say 'All Done. All Done'")


#--------------------------------------------------------
# file = hs.az_files[13]
# file = 'Datasets/Arizona/Cubes/az_cube_1 Export 1.bip'
# file = '/Volumes/KM1TB/HS Data/Agricenter Summer 22/agc_test_10b_Pika_L_71-Georectify Airborne Datacube.bip'

# image = h(file, stats=True)

# x1, x2, y1, y2 = 31, 306, 22, 470
# size = 20
# x1, x2, y1, y2 = int((image.data.shape[1]/2)-size), int((image.data.shape[1]/2)+size), int((image.data.shape[0]/2)-size), int((image.data.shape[0]/2)+size)
# bounds = [x1, x2, y1, y2]
# print(bounds)
# image = image.crop(bounds)
# image.display_RGB(display=True, save=False)

# image.export()
# image.crop_many()

# x1, x2, y1, y2 = 16, 296, 2, 470
# bounds = [x1, x2, y1, y2]
# hs.crop_export(file, bounds, 1)


#---------------------------------------------------------------------
''' MAPIR PROCESSES '''
#---------------------------------------------------------------------

# file = f'/Volumes/KM1TB/HS Data/Agricenter Summer 22/agc_test_10b_Pika_L_104-Georectify Airborne Datacube.bip'

# image = h(file, stats=False) #a hyperspectral image object using hyperspectral
# image.display_image()
# image.display_RGB(display=True)
# image.display_Mapir_Single(display=True)
# image.display_Mapir_Range(display=True)





process.stats()