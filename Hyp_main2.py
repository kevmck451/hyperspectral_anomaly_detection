#Used for Testing various aspects of the Recon, Hyperspectral, and Material Files
#Kevin McKenzie 2022

import matplotlib.pyplot as plt
import numpy as np
import random
import os

from Materials.Materials import Material as m
from Materials.Materials import Material_Lib
from Hyperspectral_1_2 import Hyperspectral as h
import Hyperspectral_1_2 as hs
from Hyperspectral_1_2 import time_class as t

process = t('Main')

# record_main = r().compile_detectors()

# Pipeline Process
# instead of iterating through a list / needs to iterate through folder
hs.preprocess(hs.AC_22)
hs.overview_all(hs.AC_22_Pro)

# file = hs.files[10]
# file = hs.avc_set[6]
# file = hs.AV_Crop_8A

# file = hs.AC_22[2]
# file = hs.AC_22_Pro[0]
# file = hs.AZ_22_Full
# file = hs.AZ_22[2]
# file = hs.AZ_22_Pro[2]


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

# image.sc_filtered(display=True, save=False)
# image.Anomaly_IF(display=True, save=False)
# image.Anomaly_RXSeg(display=True, save=False)
# image.Anomaly_RXGlobal(display=True, save=False)
# image.Anomaly_LOF_fp(display=True, overlay=True, save=False, ncomps=14, neighbor=image.img_bands)
# image.Anomaly_PCA(display=True, save=False, stats=False)
# image.Anomaly_Autoencoder(dp=True, sv=False, tv=90)

# image.overview(display=True, save=False)
# image.Anomaly_Stack_1(display=True, overlay=True, save=False)
# image.Anomaly_Stack_1b(display=False, overlay=True, save=False)
# image.Anomaly_Stack_2(display=False, overlay=False, save=False)
# image.Anomaly_Stack_3(display=True, overlay=True, save=False)

i = 1
# for file in hs.AC_22:
#     print(f'------------------ Cube {i} ------------------')
#     # file = hs.az_files[file]
#     image = h(file, stats=False)
#     image.display_RGB(display=False, save=True)
#     i+=1

os.system("say 'All Done. All Done'")

def main():

    ''' Organizing Dataset'''
    # Datasets - bringing in data in sets, not just single image
    # Renameing all files in folder with Dataset Name: Location-Date-CubeID#
    # Georectifing vs Using the cubes
    # Cropping - dealing with spatial areas


    '''Preprocessing'''
    # Preprocess Data - save the data in new folder so it only needs to be done once
    # Folder needs a txt doc with settings for preprocessing to easily remember



    '''Analysis'''
    # Vegetation Indices class:
    # Runs all indices and gives optinos for which to display
    # or runs them all and creates a report of all indicies


    # Anomaly Detection Class: What anomalies did you find?

# if __name__ == "__main__":
#     main()

process.stats()