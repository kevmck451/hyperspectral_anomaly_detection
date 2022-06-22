'''Opening a ENVI File'''

#from osgeo import gdal
from scipy import io
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import spectral

'''File Locations'''
image_file = 'Resonon/PikaL/PikaD1.bsq'
image_hdr = 'Resonon/PikaL/PikaD1.hdr'

image_file2 = 'Resonon/PikaL/PikaD2.bsq'
image_hdr2 = 'Resonon/PikaL/PikaD2.hdr'

image_file3 = '/AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img'
image_hdr3 = '/AVIRIS/f100704t01p00r06rdn_b_sc01_ort_img.hdr'

imgtif = 'IndianPines/aviris_HD/IndianPine_Site3.tif'

imgmat = 'Pavia/Cuprite_sc03.a.rfl.mat'

#-----------------------------------------------------------------------------
'''OPEN EVNI File with GDAL & Spectral'''

# img = gdal.Open(image_file3)

imgs = spectral.open_image(image_hdr3) #returns SpyFile object to access the file
#imgs2 = spectral.io.envi.open(image_hdr)
#print(imgs) #Metadata about image

# '''Open GeoTiff with GDAL'''
# imgg = gdal.Open(imgtif)
#
# '''Open Matlab File with SciPy'''
# imgt = io.loadmat(imgmat)

#-----------------------------------------------------------------------------

'''Display Images'''

view1 = spectral.imshow(imgs, bands=(30, 20, 10)) #30,20,10 are the bands that are R,G,B
# print(view1) #Metadata about image

spectral.imshow(imgs, bands=(30, 20, 10))
#plt.show()
plt.pause(60) #for script
